"""Cloud Agent Service — FastAPI wrapper around CloudAI (Groq/Llama).

Exposes the cloud threat-assessment model as an A2A-compatible HTTP service.
Accepts observation history and optional RAG context, returns a structured
decision (CLEAR / INVESTIGATE / ALERT).
"""

import sys
import time
from pathlib import Path

from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).parent.parent))

from a2a.protocol import AgentCard, AgentMessage, AgentResponse

app = FastAPI(title="Cloud AI Agent", version="1.0.0")

# Global CloudAI instance (initialized at startup)
_cloud_ai = None


def init_cloud_ai(api_key: str, model_id: str = "llama-3.3-70b-versatile"):
    """Initialize the CloudAI model. Called before starting the server."""
    global _cloud_ai
    from models.cloud_ai import CloudAI
    _cloud_ai = CloudAI(api_key=api_key, model_id=model_id)
    return _cloud_ai


@app.get("/agent-card")
def agent_card() -> AgentCard:
    """Return this agent's capability card."""
    return AgentCard(
        name="cloud_ai",
        description="Cloud threat assessment (Groq/Llama 3.3 70B) for violence detection",
        capabilities=["assess"],
        endpoints=[
            {"path": "/assess", "method": "POST",
             "msg_type": "assess",
             "description": "Assess threat level from observation history + optional RAG context"},
        ],
        metadata={
            "model_id": _cloud_ai.model_id if _cloud_ai else "not loaded",
            "timeout_seconds": 5,
            "fallback": "keyword-based edge fallback",
        },
    )


@app.post("/assess")
def assess(message: AgentMessage) -> AgentResponse:
    """Assess threat level from observation history.

    Payload:
        history (List[str]): Timestamped observation log
        rag_context (List[dict], optional): Similar past cases from RAG

    Returns:
        status (str): "CLEAR", "INVESTIGATE", or "ALERT"
        confidence (int): 0-100
        question (str): Follow-up question (if INVESTIGATE)
        reason (str): Decision rationale
    """
    start = time.time()

    try:
        history = message.payload["history"]
        rag_context = message.payload.get("rag_context", None)

        decision = _cloud_ai.assess_threat(history, rag_context=rag_context)

        return AgentResponse(
            status="success",
            payload=decision,
            processing_time_ms=round((time.time() - start) * 1000, 2),
            correlation_id=message.correlation_id,
        )
    except Exception as e:
        return AgentResponse(
            status="error",
            payload={
                "status": "CLEAR",
                "confidence": 0,
                "question": "",
                "reason": f"Cloud agent error: {str(e)}",
            },
            processing_time_ms=round((time.time() - start) * 1000, 2),
            correlation_id=message.correlation_id,
            error_detail=str(e),
        )


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": _cloud_ai is not None,
    }
