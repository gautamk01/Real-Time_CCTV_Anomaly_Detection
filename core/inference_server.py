"""GPU Inference Server — Thread-safe GPU access for multi-camera pipelines.

Provides a single-model, priority-queued inference server that allows
multiple camera pipelines to share one EdgeVision model instance without
VRAM duplication. GPU requests are serialized through a priority queue
and processed by a dedicated worker thread.

Architecture:
    Camera Pipeline 1 ──┐
    Camera Pipeline 2 ──┤──▶ PriorityQueue ──▶ GPU Worker Thread ──▶ EdgeVision
    Camera Pipeline N ──┘        (FIFO + priority)         (single model)

Each submit_*() call returns a Future that resolves when the GPU finishes.
"""

import threading
import time
import traceback
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import IntEnum
from queue import PriorityQueue
from typing import Any, Optional

from PIL import Image


class RequestPriority(IntEnum):
    """GPU request priority levels (lower = higher priority)."""
    ACTIVE_INVESTIGATION = 0   # Encode/answer for an in-progress investigation
    PRE_ENCODE = 1             # Speculative pre-encoding during cloud latency
    BACKGROUND = 2             # Low-priority background tasks


@dataclass(order=True)
class InferenceRequest:
    """A single GPU inference request submitted to the server."""
    priority: int
    sequence: int = field(compare=True)  # FIFO within same priority
    operation: str = field(compare=False)
    payload: dict = field(compare=False)
    camera_id: str = field(compare=False, default="")
    future: Future = field(compare=False, default_factory=Future)


class InferenceServer:
    """Thread-safe GPU inference server backing multiple camera pipelines.

    Owns a single EdgeVision model instance and processes encode/answer
    requests through a priority queue. Callers receive Futures that
    resolve when the GPU completes their request.

    Usage:
        server = InferenceServer(edge_vision)
        server.start()

        # From any camera thread:
        future = server.submit_encode(frame, priority=RequestPriority.ACTIVE_INVESTIGATION)
        encoded = future.result()  # blocks until GPU processes it

        future = server.submit_answer(encoded, question)
        answer = future.result()

        server.shutdown()
    """

    _SENTINEL = "SHUTDOWN"

    def __init__(self, edge_vision, max_queue_size: int = 32):
        """
        Args:
            edge_vision: EdgeVision model instance (single copy, shared).
            max_queue_size: Maximum pending requests before blocking.
        """
        self.edge = edge_vision
        self._queue: PriorityQueue[InferenceRequest] = PriorityQueue(
            maxsize=max_queue_size,
        )
        self._sequence = 0
        self._seq_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._stats_lock = threading.Lock()
        self._stats = {
            "requests_processed": 0,
            "encode_calls": 0,
            "answer_calls": 0,
            "analyze_calls": 0,
            "errors": 0,
            "total_gpu_time_ms": 0.0,
        }

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._sequence += 1
            return self._sequence

    # ── Public submission API ────────────────────────────────────────

    def submit_encode(
        self,
        image: Image.Image,
        priority: int = RequestPriority.ACTIVE_INVESTIGATION,
        camera_id: str = "",
    ) -> Future:
        """Submit an image encoding request to the GPU queue.

        Args:
            image: PIL Image to encode.
            priority: Request priority (lower = processed sooner).
            camera_id: Originating camera (for logging).

        Returns:
            Future that resolves to the encoded image embeddings.
        """
        req = InferenceRequest(
            priority=priority,
            sequence=self._next_seq(),
            operation="encode",
            payload={"image": image},
            camera_id=camera_id,
        )
        self._queue.put(req)
        return req.future

    def submit_answer(
        self,
        encoded_image: Any,
        question: str,
        max_tokens: Optional[int] = None,
        priority: int = RequestPriority.ACTIVE_INVESTIGATION,
        camera_id: str = "",
    ) -> Future:
        """Submit a question-answering request using a pre-encoded image.

        Args:
            encoded_image: Result from a previous encode() call.
            question: Question to ask about the image.
            max_tokens: Max tokens for the answer.
            priority: Request priority.
            camera_id: Originating camera.

        Returns:
            Future that resolves to the answer string.
        """
        req = InferenceRequest(
            priority=priority,
            sequence=self._next_seq(),
            operation="answer",
            payload={
                "encoded_image": encoded_image,
                "question": question,
                "max_tokens": max_tokens,
            },
            camera_id=camera_id,
        )
        self._queue.put(req)
        return req.future

    def submit_analyze(
        self,
        image: Image.Image,
        question: str,
        max_tokens: Optional[int] = None,
        priority: int = RequestPriority.ACTIVE_INVESTIGATION,
        camera_id: str = "",
    ) -> Future:
        """Submit combined encode+answer request.

        Convenience method that performs both steps atomically on the GPU
        thread (no queue gap between encode and answer).

        Args:
            image: PIL Image to analyze.
            question: Question to ask.
            max_tokens: Token budget.
            priority: Request priority.
            camera_id: Originating camera.

        Returns:
            Future that resolves to the answer string.
        """
        req = InferenceRequest(
            priority=priority,
            sequence=self._next_seq(),
            operation="analyze",
            payload={
                "image": image,
                "question": question,
                "max_tokens": max_tokens,
            },
            camera_id=camera_id,
        )
        self._queue.put(req)
        return req.future

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the GPU worker thread."""
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(
            target=self._gpu_worker_loop,
            daemon=True,
            name="gpu-inference-server",
        )
        self._worker.start()
        print("[GPU SERVER] Inference server started")

    def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully shutdown the GPU worker."""
        if not self._running:
            return
        self._running = False
        # Send sentinel to unblock the worker
        sentinel = InferenceRequest(
            priority=999,
            sequence=self._next_seq(),
            operation=self._SENTINEL,
            payload={},
        )
        self._queue.put(sentinel)
        if self._worker:
            self._worker.join(timeout=timeout)
        print("[GPU SERVER] Inference server stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def pending_requests(self) -> int:
        return self._queue.qsize()

    def get_stats(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)

    # ── GPU Worker Loop ──────────────────────────────────────────────

    def _gpu_worker_loop(self) -> None:
        """Main loop: dequeue requests and execute on GPU."""
        while self._running:
            try:
                req = self._queue.get(timeout=0.1)
            except Exception:
                continue

            if req.operation == self._SENTINEL:
                break

            t0 = time.perf_counter()
            try:
                result = self._execute_request(req)
                req.future.set_result(result)
            except Exception as exc:
                req.future.set_exception(exc)
                with self._stats_lock:
                    self._stats["errors"] += 1
                print(
                    f"[GPU SERVER] Error processing {req.operation} "
                    f"for {req.camera_id}: {exc}"
                )
                traceback.print_exc()
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                with self._stats_lock:
                    self._stats["requests_processed"] += 1
                    self._stats["total_gpu_time_ms"] += elapsed_ms

    def _execute_request(self, req: InferenceRequest) -> Any:
        """Dispatch a request to the appropriate EdgeVision method."""
        op = req.operation
        payload = req.payload

        if op == "encode":
            with self._stats_lock:
                self._stats["encode_calls"] += 1
            return self.edge.encode(payload["image"])

        elif op == "answer":
            with self._stats_lock:
                self._stats["answer_calls"] += 1
            return self.edge.answer(
                payload["encoded_image"],
                payload["question"],
                max_tokens=payload.get("max_tokens"),
            )

        elif op == "analyze":
            with self._stats_lock:
                self._stats["analyze_calls"] += 1
            return self.edge.analyze(
                payload["image"],
                payload["question"],
                max_tokens=payload.get("max_tokens"),
            )

        else:
            raise ValueError(f"Unknown operation: {op}")
