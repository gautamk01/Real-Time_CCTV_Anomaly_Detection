"""Edge Agent Service — FastAPI wrapper around EdgeVision (Moondream2).

Exposes the local vision model as an A2A-compatible HTTP service with
endpoints for frame observation, encoding, and question answering.

The encode/answer two-phase API is preserved for pipeline parallelism:
the client can call /encode and /assess (on cloud) in parallel, then
/answer once the cloud responds with a follow-up question.

Encoded images are stored server-side keyed by encode_id since the
opaque EncodedImage object cannot be serialized over HTTP.
"""

import base64
import gc
import io
import sys
import time
import threading
from pathlib import Path
from typing import Optional
from uuid import uuid4

import torch

from fastapi import FastAPI
from PIL import Image

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from a2a.protocol import AgentCard, AgentMessage, AgentResponse

app = FastAPI(title="Edge Vision Agent", version="1.0.0")

# --- Server-side state for encoded images ---
_encoded_cache: dict = {}  # {encode_id: (encoded_image, timestamp)}
_cache_lock = threading.Lock()
_CACHE_TTL_SECONDS = 30  # Auto-expire after 30s

# Global EdgeVision instance (initialized at startup)
_edge_vision = None


def _cleanup_expired():
    """Remove expired encode cache entries."""
    now = time.time()
    with _cache_lock:
        expired = [k for k, (_, ts) in _encoded_cache.items()
                   if now - ts > _CACHE_TTL_SECONDS]
        for k in expired:
            del _encoded_cache[k]


def _reclaim_gpu_memory():
    """Proactively free GPU memory after inference.

    Runs Python GC to drop any lingering tensor references, then tells
    PyTorch's CUDA caching allocator to release freed blocks back to CUDA.
    This prevents memory accumulation during rapid successive investigations.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _decode_image(image_b64: str) -> Image.Image:
    """Decode a base64 string to PIL Image."""
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _encode_image(image: Image.Image) -> str:
    """Encode a PIL Image to base64 string."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def init_edge_vision(
    model_id: str = "vikhyatk/moondream2",
    device: str = "auto",
    quant_mode: str = "auto",
):
    """Initialize the EdgeVision model. Called before starting the server."""
    global _edge_vision
    from models.edge_vision import EdgeVision
    _edge_vision = EdgeVision(
        model_id=model_id,
        device=device,
        quant_mode=quant_mode,
    )
    return _edge_vision


@app.get("/agent-card")
def agent_card() -> AgentCard:
    """Return this agent's capability card."""
    metadata = {}
    if _edge_vision:
        metadata = _edge_vision.get_device_info()

    return AgentCard(
        name="edge_vision",
        description="Local vision model (Moondream2) for real-time frame analysis",
        capabilities=["observe", "encode", "answer"],
        endpoints=[
            {"path": "/observe", "method": "POST",
             "msg_type": "observe",
             "description": "Full analyze: encode + answer in one call"},
            {"path": "/encode", "method": "POST",
             "msg_type": "encode",
             "description": "Phase 1: Encode image into embeddings (heavy GPU)"},
            {"path": "/answer", "method": "POST",
             "msg_type": "answer",
             "description": "Phase 2: Answer question using pre-encoded image (lightweight)"},
        ],
        metadata=metadata,
    )


@app.post("/observe")
def observe(message: AgentMessage) -> AgentResponse:
    """Full observation: encode image + answer question in one call.

    Payload:
        image_b64 (str): Base64-encoded image
        question (str): Question to ask about the image

    Returns:
        description (str): Answer from the vision model
    """
    start = time.time()
    _cleanup_expired()

    try:
        image = _decode_image(message.payload["image_b64"])
        question = message.payload["question"]
        max_tokens = message.payload.get("max_tokens")

        description = _edge_vision.analyze(image, question, max_tokens=max_tokens)

        # Proactive cleanup: clear encoding cache and free CUDA blocks
        _edge_vision.clear_cache()
        _reclaim_gpu_memory()

        return AgentResponse(
            status="success",
            payload={"description": description},
            processing_time_ms=round((time.time() - start) * 1000, 2),
            correlation_id=message.correlation_id,
        )
    except Exception as e:
        _reclaim_gpu_memory()
        return AgentResponse(
            status="error",
            payload={},
            processing_time_ms=round((time.time() - start) * 1000, 2),
            correlation_id=message.correlation_id,
            error_detail=str(e),
        )


@app.post("/encode")
def encode(message: AgentMessage) -> AgentResponse:
    """Phase 1: Encode image into vision embeddings (heavy GPU work).

    The encoded image is stored server-side and an encode_id is returned.
    Use this encode_id with /answer for the lightweight Phase 2 step.

    Payload:
        image_b64 (str): Base64-encoded image

    Returns:
        encode_id (str): Reference ID for the encoded image
    """
    start = time.time()
    _cleanup_expired()

    try:
        image = _decode_image(message.payload["image_b64"])
        encoded_image = _edge_vision.encode(image)

        encode_id = str(uuid4())
        with _cache_lock:
            _encoded_cache[encode_id] = (encoded_image, time.time())

        return AgentResponse(
            status="success",
            payload={"encode_id": encode_id},
            processing_time_ms=round((time.time() - start) * 1000, 2),
            correlation_id=message.correlation_id,
        )
    except Exception as e:
        return AgentResponse(
            status="error",
            payload={},
            processing_time_ms=round((time.time() - start) * 1000, 2),
            correlation_id=message.correlation_id,
            error_detail=str(e),
        )


@app.post("/answer")
def answer(message: AgentMessage) -> AgentResponse:
    """Phase 2: Answer question using a pre-encoded image (lightweight step).

    Payload:
        encode_id (str): Reference ID from a previous /encode call
        question (str): Question to ask about the image

    Returns:
        description (str): Answer from the vision model
    """
    start = time.time()

    try:
        encode_id = message.payload["encode_id"]
        question = message.payload["question"]
        max_tokens = message.payload.get("max_tokens")

        with _cache_lock:
            entry = _encoded_cache.get(encode_id)

        if entry is None:
            return AgentResponse(
                status="error",
                payload={},
                processing_time_ms=round((time.time() - start) * 1000, 2),
                correlation_id=message.correlation_id,
                error_detail=f"encode_id '{encode_id}' not found or expired",
            )

        encoded_image, _ = entry
        description = _edge_vision.answer(
            encoded_image,
            question,
            max_tokens=max_tokens,
        )

        # Delete used entry immediately — encode/answer is one-shot
        with _cache_lock:
            _encoded_cache.pop(encode_id, None)
        _reclaim_gpu_memory()

        return AgentResponse(
            status="success",
            payload={"description": description},
            processing_time_ms=round((time.time() - start) * 1000, 2),
            correlation_id=message.correlation_id,
        )
    except Exception as e:
        # Still try to clean up on error
        with _cache_lock:
            _encoded_cache.pop(message.payload.get("encode_id", ""), None)
        _reclaim_gpu_memory()
        return AgentResponse(
            status="error",
            payload={},
            processing_time_ms=round((time.time() - start) * 1000, 2),
            correlation_id=message.correlation_id,
            error_detail=str(e),
        )


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": _edge_vision is not None,
        "cache_size": len(_encoded_cache),
    }
