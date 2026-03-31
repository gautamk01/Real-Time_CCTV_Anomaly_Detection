"""AI Investigator - Real-Time Threat Analysis with Pipeline Parallelism.

Supports two GPU access modes:
1. Direct mode: edge_vision=EdgeVision instance (legacy, single-camera)
2. Server mode: inference_server=InferenceServer (distributed, multi-camera)

When using InferenceServer, all GPU calls go through the shared queue,
allowing multiple investigators to share one model without VRAM duplication.
"""

import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Tuple, Optional, Dict, List
from PIL import Image


@dataclass
class PreEncodedFrame:
    """A frame that has been pre-encoded by the VLM during cloud latency."""
    image: Image.Image
    timestamp: str
    encoded: Any  # Opaque EncodedImage from edge.encode()
    encode_time: float  # Wall-clock time when encoded


class PreEncodeQueue:
    """Thread-safe queue of pre-encoded frames for pipeline parallelism.

    During cloud API latency (1-2s), the VLM continuously encodes fresh
    frames into this queue. When the cloud responds, the investigator
    pops the most recent encoded frame for immediate use.
    """

    def __init__(self, maxlen: int = 3):
        self._queue: deque[PreEncodedFrame] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, frame: PreEncodedFrame) -> None:
        with self._lock:
            self._queue.append(frame)

    def pop_latest(self) -> Optional[PreEncodedFrame]:
        """Pop and return the most recent pre-encoded frame, discarding older ones."""
        with self._lock:
            if not self._queue:
                return None
            latest = self._queue[-1]
            self._queue.clear()
            return latest

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()


class AIInvestigator:
    """
    Real-time AI Investigator with threat-triggered conversation loop.

    Flow:
    1. Motion detected -> Edge analyzes FIRST frame -> sends to Cloud
    2. Cloud says CLEAR -> back to monitoring
    3. Cloud says INVESTIGATE -> Edge grabs CURRENT frame -> analyzes -> sends to Cloud
    4. Loop continues until Cloud says CLEAR, REVIEW, or ALERT

    Pipeline Optimization:
    While Cloud processes round N, we pre-encode the next frame on the GPU.
    When Cloud returns with round N+1's question, we only need the lightweight
    answer() step instead of the full encode+answer cycle. This hides 200-500ms
    of GPU encoding time behind Cloud API latency.

    GPU Access Modes:
    - Direct: self.edge is an EdgeVision instance (calls model directly)
    - Server: self.gpu_server is an InferenceServer (submits to shared queue)
    """

    INITIAL_EDGE_QUESTION = (
        "Describe only directly visible actions. Mention physical contact, "
        "falls, weapons, injuries, or say unclear if visibility is poor."
    )
    FOLLOWUP_PREFIX = "Answer only from visible evidence:"

    def __init__(self, edge_vision=None, cloud_ai=None, frame_provider: Callable = None,
                 max_rounds: int = 3, fps: float = 30.0,
                 initial_edge_max_tokens: int = 24,
                 followup_edge_max_tokens: int = 16,
                 save_alerts: bool = True, alerts_dir: Path = Path("alerts"),
                 buffer_duration: int = 10, fcm_notifier=None,
                 metrics_logger=None, a2a_client=None,
                 inference_server=None, camera_id: str = ""):
        """
        Initialize AI investigator.

        Args:
            edge_vision: EdgeVision instance (direct GPU mode — legacy).
            cloud_ai: CloudAI instance.
            frame_provider: Callback that returns (pil_image, timestamp).
            max_rounds: Maximum investigation rounds.
            fps: Video FPS (for buffer size calculation).
            save_alerts: Whether to save frames on ALERT/REVIEW.
            alerts_dir: Directory to save incidents.
            buffer_duration: Seconds of frames to buffer.
            fcm_notifier: Optional FCMNotifier for push notifications.
            metrics_logger: Optional MetricsLogger for benchmarking.
            a2a_client: Optional A2AClient for networked agents.
            inference_server: Optional InferenceServer for distributed mode.
                             When provided, GPU calls go through the shared queue
                             instead of direct EdgeVision calls.
            camera_id: Camera identifier (used for GPU request tagging).
        """
        self.edge = edge_vision
        self.cloud = cloud_ai
        self.get_frame = frame_provider
        self.max_rounds = max_rounds
        self.initial_edge_max_tokens = initial_edge_max_tokens
        self.followup_edge_max_tokens = followup_edge_max_tokens
        self.fcm_notifier = fcm_notifier
        self.metrics = metrics_logger
        self.a2a = a2a_client

        # GPU access mode
        self.gpu_server = inference_server
        self._camera_id = camera_id

        # Alert saving
        self.save_alerts = save_alerts
        self.alerts_dir = Path(alerts_dir)

        if self.save_alerts:
            self.alerts_dir.mkdir(parents=True, exist_ok=True)

        # Circular buffer for frame history
        buffer_size = int(fps * buffer_duration)
        self.frame_buffer = deque(maxlen=buffer_size)

        mode = "server" if self.gpu_server else ("a2a" if self.a2a else "direct")
        print(f"✅ [AI] Investigator ready (camera={camera_id}, mode={mode})")

    # ── GPU Access Helpers ───────────────────────────────────────────

    def _gpu_encode(self, image: Image.Image, priority: int = 0) -> Any:
        """Encode image via server queue or direct call."""
        if self.a2a:
            return self.a2a.encode(image)
        if self.gpu_server:
            future = self.gpu_server.submit_encode(
                image, priority=priority, camera_id=self._camera_id,
            )
            return future.result()
        return self.edge.encode(image)

    def _gpu_answer(
        self, encoded_image: Any, question: str,
        max_tokens: Optional[int] = None, priority: int = 0,
    ) -> str:
        """Answer question via server queue or direct call."""
        if self.a2a:
            return self.a2a.answer(encoded_image, question, max_tokens=max_tokens)
        if self.gpu_server:
            future = self.gpu_server.submit_answer(
                encoded_image, question, max_tokens=max_tokens,
                priority=priority, camera_id=self._camera_id,
            )
            return future.result()
        return self.edge.answer(encoded_image, question, max_tokens=max_tokens)

    def _gpu_analyze(
        self, image: Image.Image, question: str,
        max_tokens: Optional[int] = None, priority: int = 0,
    ) -> str:
        """Combined encode+answer via server queue or direct call."""
        if self.a2a:
            return self.a2a.observe(image, question, max_tokens=max_tokens)
        if self.gpu_server:
            future = self.gpu_server.submit_analyze(
                image, question, max_tokens=max_tokens,
                priority=priority, camera_id=self._camera_id,
            )
            return future.result()
        return self.edge.analyze(image, question, max_tokens=max_tokens)

    def _gpu_encode_async(self, image: Image.Image, priority: int = 1) -> Optional[Future]:
        """Non-blocking encode — returns Future (server mode) or None (direct/a2a)."""
        if self.gpu_server:
            return self.gpu_server.submit_encode(
                image, priority=priority, camera_id=self._camera_id,
            )
        return None

    # ── Frame buffering ──────────────────────────────────────────────

    def buffer_frame(self, frame: Image.Image, timestamp: str, motion: bool = False):
        """Add frame to circular buffer."""
        self.frame_buffer.append({
            'frame': frame,
            'timestamp': timestamp,
            'motion': motion
        })

    def _save_incident(self, verdict: Dict, history: List[str], prefix: str) -> Optional[str]:
        """Persist buffered frames for later review."""
        if not self.save_alerts or len(self.frame_buffer) == 0:
            return None

        from .alert_saver import save_alert_incident

        frames_data = list(self.frame_buffer)
        return save_alert_incident(
            frames_data,
            self.alerts_dir,
            verdict,
            history,
            prefix=prefix,
        )

    # ── Cloud access helper ──────────────────────────────────────────

    def _cloud_assess(self, history: List[str], rag_context=None) -> Dict:
        """Cloud assessment via A2A or direct call."""
        if self.a2a:
            return self.a2a.assess(history, rag_context)
        return self.cloud.assess_threat(history, rag_context)

    # ── Investigation Loop ───────────────────────────────────────────

    def investigate_realtime(
        self,
        initial_frame: Image.Image,
        initial_timestamp: str,
        queue_wait_ms: Optional[float] = None,
        queue_depth: Optional[int] = None,
    ) -> Dict:
        """
        Real-time threat-triggered investigation loop with pipeline parallelism.

        Pipeline optimization: While the cloud API processes round N, we
        pre-encode the next frame on the GPU. This overlaps GPU compute
        with network I/O, saving 200-500ms per round.

        Returns:
            Dict with investigation results.
        """
        cam_tag = f"[{self._camera_id}]" if self._camera_id else ""
        print(f"\n{'='*60}")
        print(f"🔍 {cam_tag} [INVESTIGATION] Started at {initial_timestamp}")
        print(f"{'='*60}")

        history: List[str] = []
        asked_questions: List[str] = []  # Anti-repetition: track questions per investigation

        # Start metrics tracking
        inv_id = None
        if self.metrics:
            video_file = getattr(self, '_video_file', 'unknown')
            inv_id = self.metrics.start_investigation(
                video_file,
                initial_timestamp,
                camera_id=self._camera_id or None,
                queue_wait_ms=queue_wait_ms,
                queue_depth=queue_depth,
            )
            self.metrics.sample_system_metrics()

        # Buffer the initial frame
        self.buffer_frame(initial_frame, initial_timestamp, motion=True)

        # STEP 1: Initial analysis — full encode + answer
        print(f"\n📸 {cam_tag} [EDGE @{initial_timestamp}] Analyzing initial frame...")
        start_time = time.time()

        initial_question = self.INITIAL_EDGE_QUESTION
        initial_desc = self._gpu_analyze(
            initial_frame, initial_question,
            max_tokens=self.initial_edge_max_tokens,
        )
        edge_time = time.time() - start_time

        print(f"   Response: {initial_desc} (⏱️ {edge_time:.2f}s)")
        history.append(f"[{initial_timestamp}] Initial scan: {initial_desc}")

        if self.metrics and inv_id:
            self.metrics.record_edge_call(inv_id, edge_time, initial_desc)

        # RAG retrieval
        rag_context = None
        if self.a2a and self.a2a.has_rag:
            rag_context = self.a2a.retrieve(" ".join(history), limit=3)
            if rag_context:
                print(f"   📚 [RAG] Found {len(rag_context)} similar past cases")

        # STEP 2: Cloud assessment + parallel pre-encode
        print(f"\n☁️  {cam_tag} [CLOUD] Analyzing initial report...")
        cloud_start = time.time()

        pre_queue = PreEncodeQueue(maxlen=3)

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Inject already-asked questions so cloud is forced to ask something new
            history_with_context = list(history)
            if asked_questions:
                history_with_context.append(
                    f"[META] Questions already asked (do NOT repeat): "
                    + " | ".join(f'"{q}"' for q in asked_questions)
                )
            cloud_future = executor.submit(self._cloud_assess, history_with_context, rag_context)

            # Pre-encode frames while cloud is thinking
            while not cloud_future.done():
                next_frame, next_ts = self.get_frame()
                if next_frame is not None:
                    try:
                        enc = self._gpu_encode(next_frame, priority=1)
                        pre_queue.push(PreEncodedFrame(
                            image=next_frame, timestamp=next_ts,
                            encoded=enc, encode_time=time.time(),
                        ))
                    except Exception as e:
                        print(f"   [PRE-ENCODE] skipped frame: {e}")
                        break
                time.sleep(0.01)

            decision = cloud_future.result()

        status = decision.get('status', 'CLEAR')
        confidence = decision.get('confidence', 0)
        reason = decision.get('reason', '')

        cloud_time = time.time() - cloud_start

        print(f"   Status: {status} | Confidence: {confidence}% (⏱️ {cloud_time:.2f}s)")
        print(f"   Reason: {reason}")

        if self.metrics and inv_id:
            prompt_len = len("\n".join(history))
            self.metrics.record_cloud_call(inv_id, cloud_time, decision, prompt_len)

        # STEP 3: Investigation loop with pipeline parallelism
        round_num = 1
        while status == "INVESTIGATE" and round_num < self.max_rounds:
            round_num += 1
            round_start = time.time()

            question = decision.get('question', 'Describe any aggressive actions.')
            # Track this question to prevent repetition in next round
            if question:
                asked_questions.append(question)
            edge_question = f"{self.FOLLOWUP_PREFIX} {question}"
            print(f"\n☁️  {cam_tag} [CLOUD] Asking: \"{question}\"")

            # Use pre-encoded frame if available
            pre = pre_queue.pop_latest()
            if pre is not None:
                current_frame = pre.image
                current_ts = pre.timestamp
                enc_image = pre.encoded
                print(f"📸 {cam_tag} [EDGE @{current_ts}] Answering with pre-encoded frame...")
            else:
                current_frame, current_ts = self.get_frame()
                if current_frame is None:
                    print(f"   ⚠️  [EDGE] No frame available, waiting...")
                    time.sleep(0.5)
                    current_frame, current_ts = self.get_frame()
                    if current_frame is None:
                        print(f"   ❌ [EDGE] Still no frame, aborting investigation")
                        break
                print(f"📸 {cam_tag} [EDGE @{current_ts}] Encoding + answering CURRENT frame...")
                enc_image = self._gpu_encode(current_frame)

            self.buffer_frame(current_frame, current_ts)

            # Phase 2: Answer question
            answer = self._gpu_answer(
                enc_image, edge_question,
                max_tokens=self.followup_edge_max_tokens,
            )

            edge_round_time = time.time() - round_start
            print(f"   Response: {answer} (⏱️ {edge_round_time:.2f}s)")

            if self.metrics and inv_id:
                self.metrics.record_edge_call(inv_id, edge_round_time, answer)

            history.append(f"[{current_ts}] Q: {question}")
            history.append(f"[{current_ts}] A: {answer}")

            # Cloud assessment + parallel pre-encode for next round
            print(f"\n☁️  {cam_tag} [CLOUD] Analyzing update (Round {round_num})...")
            cloud_start = time.time()

            pre_queue.clear()

            with ThreadPoolExecutor(max_workers=2) as executor:
                # Inject anti-repetition context for each subsequent cloud call
                history_with_context = list(history)
                if asked_questions:
                    history_with_context.append(
                        f"[META] Questions already asked (do NOT repeat): "
                        + " | ".join(f'"{q}"' for q in asked_questions)
                    )
                cloud_future = executor.submit(self._cloud_assess, history_with_context, rag_context)

                if round_num < self.max_rounds:
                    while not cloud_future.done():
                        next_frame, next_ts = self.get_frame()
                        if next_frame is not None:
                            try:
                                enc = self._gpu_encode(next_frame, priority=1)
                                pre_queue.push(PreEncodedFrame(
                                    image=next_frame, timestamp=next_ts,
                                    encoded=enc, encode_time=time.time(),
                                ))
                            except Exception as e:
                                print(f"   [PRE-ENCODE] skipped frame: {e}")
                                break
                        time.sleep(0.01)

                decision = cloud_future.result()

            status = decision.get('status', 'CLEAR')
            confidence = decision.get('confidence', 0)
            reason = decision.get('reason', '')

            cloud_round_time = time.time() - cloud_start
            total_round_time = time.time() - round_start

            print(f"   Status: {status} | Confidence: {confidence}% (⏱️ {cloud_round_time:.2f}s)")
            print(f"   Reason: {reason}")
            print(f"   ⚡ Round {round_num} total: {total_round_time:.2f}s")

            if self.metrics and inv_id:
                prompt_len = len("\n".join(history))
                self.metrics.record_cloud_call(inv_id, cloud_round_time, decision, prompt_len)
                self.metrics.sample_system_metrics()

        # FINAL VERDICT
        print(f"\n{'='*60}")
        if status == "ALERT":
            print(f"🚨🚨🚨 {cam_tag} [VERDICT] VIOLENCE DETECTED! 🚨🚨🚨")
            print(f"   Confidence: {confidence}%")
            print(f"   Reason: {reason}")
            print(f"   Rounds of investigation: {round_num}")

            alert_dir_name = self._save_incident(decision, history, prefix="alert")

            if self.fcm_notifier:
                self.fcm_notifier.send_alert(
                    verdict=decision,
                    history=history,
                    timestamp=initial_timestamp,
                    alert_dir=alert_dir_name
                )

        elif status == "CLEAR":
            print(f"✅ {cam_tag} [VERDICT] ALL CLEAR - No threat detected")
            print(f"   Confidence: {confidence}%")
            print(f"   Reason: {reason}")
        elif status in ("INVESTIGATE", "REVIEW"):
            if status == "INVESTIGATE":
                status = "REVIEW"
                reason = (
                    f"{reason} (unresolved after "
                    f"{self.max_rounds} rounds)"
                )

            review_verdict = {
                "status": status,
                "confidence": confidence,
                "reason": reason,
            }

            # REVIEW is an internal signal → feeds escalation tracker.
            # No "MANUAL REVIEW" banner — the 2-agent system + temporal
            # escalation is the replacement for human review.
            print(f"🔍 {cam_tag} [MONITORING] Scene unclear — tracking for escalation")
            print(f"   Confidence: {confidence}% | Rounds: {round_num}")
            print(f"   Reason: {reason}")

            # Still save evidence frames (escalation may reference them)
            self._save_incident(
                review_verdict, history, prefix="review",
            )

        else:
            print(f"⚠️  {cam_tag} [VERDICT] Investigation ended unexpectedly")
            print(f"   Last status: {status}")
        print(f"{'='*60}\n")

        # Finalize metrics
        if self.metrics and inv_id:
            self.metrics.end_investigation(inv_id, status, confidence, round_num)
            self.metrics.sample_system_metrics()

        # Auto-ingest into RAG
        if self.a2a and self.a2a.has_rag:
            inv_unique_id = f"inv_{initial_timestamp}_{int(time.time())}"
            verdict_dict = {"status": status, "confidence": confidence, "reason": reason}
            threading.Thread(
                target=self.a2a.ingest,
                args=(history, verdict_dict, inv_unique_id),
                kwargs={"metadata": {"timestamp": initial_timestamp, "rounds": round_num}},
                daemon=True,
            ).start()
            print(f"   📚 [RAG] Ingesting investigation into knowledge base...")

        return {
            "status": status,
            "confidence": confidence,
            "reason": reason,
            "history": history,
            "rounds": round_num,
            "timestamp": initial_timestamp
        }
