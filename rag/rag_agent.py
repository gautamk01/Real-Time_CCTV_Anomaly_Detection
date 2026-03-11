"""RAG Agent Service — FastAPI wrapper around the KnowledgeBase.

Exposes retrieval and ingestion of past investigations as an
A2A-compatible HTTP service.
"""

import sys
import time
from pathlib import Path

from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).parent.parent))

from a2a.protocol import AgentCard, AgentMessage, AgentResponse

app = FastAPI(title="RAG Agent", version="1.0.0")

# Global KnowledgeBase instance (initialized at startup)
_knowledge_base = None


def init_rag(persist_dir: str = "rag_db",
             embedding_model: str = "all-MiniLM-L6-v2"):
    """Initialize the RAG KnowledgeBase. Called before starting the server."""
    global _knowledge_base
    from rag.knowledge_base import KnowledgeBase
    _knowledge_base = KnowledgeBase(
        persist_dir=persist_dir,
        embedding_model=embedding_model,
    )
    return _knowledge_base


@app.get("/agent-card")
def agent_card() -> AgentCard:
    """Return this agent's capability card."""
    return AgentCard(
        name="rag_retriever",
        description="RAG knowledge base for retrieving similar past investigations",
        capabilities=["retrieve", "ingest"],
        endpoints=[
            {"path": "/retrieve", "method": "POST",
             "msg_type": "retrieve",
             "description": "Find similar past investigations by semantic search"},
            {"path": "/ingest", "method": "POST",
             "msg_type": "ingest",
             "description": "Add a completed investigation to the knowledge base"},
            {"path": "/count", "method": "GET",
             "msg_type": "info",
             "description": "Get the number of stored investigations"},
        ],
        metadata={
            "vector_db": "ChromaDB",
            "embedding_model": _knowledge_base.embedding_model_name if _knowledge_base else "not loaded",
            "case_count": _knowledge_base.count() if _knowledge_base else 0,
        },
    )


@app.post("/retrieve")
def retrieve(message: AgentMessage) -> AgentResponse:
    """Retrieve similar past investigations.

    Payload:
        query_text (str): Text to find similar cases for
        limit (int, optional): Max results (default: 3)
        min_confidence (int, optional): Minimum confidence filter (default: 0)

    Returns:
        cases (List[dict]): Similar cases with history, metadata, similarity score
    """
    start = time.time()

    try:
        query_text = message.payload["query_text"]
        limit = message.payload.get("limit", 3)
        min_confidence = message.payload.get("min_confidence", 0)

        cases = _knowledge_base.retrieve_similar(
            query_text=query_text,
            n_results=limit,
            min_confidence=min_confidence,
        )

        return AgentResponse(
            status="success",
            payload={"cases": cases, "count": len(cases)},
            processing_time_ms=round((time.time() - start) * 1000, 2),
            correlation_id=message.correlation_id,
        )
    except Exception as e:
        return AgentResponse(
            status="error",
            payload={"cases": []},
            processing_time_ms=round((time.time() - start) * 1000, 2),
            correlation_id=message.correlation_id,
            error_detail=str(e),
        )


@app.post("/ingest")
def ingest(message: AgentMessage) -> AgentResponse:
    """Ingest a completed investigation into the knowledge base.

    Payload:
        history (List[str]): Investigation observation history
        verdict (dict): Final decision (status, confidence, reason)
        investigation_id (str): Unique ID for this investigation
        metadata (dict, optional): Extra metadata

    Returns:
        status: "ingested"
        count: New total count
    """
    start = time.time()

    try:
        _knowledge_base.add_investigation(
            history=message.payload["history"],
            verdict=message.payload["verdict"],
            investigation_id=message.payload["investigation_id"],
            metadata=message.payload.get("metadata"),
        )

        return AgentResponse(
            status="success",
            payload={"status": "ingested", "count": _knowledge_base.count()},
            processing_time_ms=round((time.time() - start) * 1000, 2),
            correlation_id=message.correlation_id,
        )
    except Exception as e:
        return AgentResponse(
            status="error",
            payload={"status": "failed"},
            processing_time_ms=round((time.time() - start) * 1000, 2),
            correlation_id=message.correlation_id,
            error_detail=str(e),
        )


@app.get("/count")
def count():
    """Return the number of investigations in the knowledge base."""
    return {"count": _knowledge_base.count() if _knowledge_base else 0}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "kb_loaded": _knowledge_base is not None,
        "case_count": _knowledge_base.count() if _knowledge_base else 0,
    }
