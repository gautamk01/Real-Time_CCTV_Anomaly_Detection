"""RAG Knowledge Base — ChromaDB-backed store for past investigations.

Uses sentence-transformers for embedding and ChromaDB for vector storage.
The embedding model runs on CPU only to avoid competing with the edge
model (Moondream2) for GPU memory.

Typical latency: <10ms for retrieval on collections under 10K documents.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer


class KnowledgeBase:
    """Vector store for violence detection investigation history.

    Stores past investigations as embedded documents and retrieves
    semantically similar cases for few-shot prompting.
    """

    def __init__(self, persist_dir: str = "rag_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Args:
            persist_dir: Directory for ChromaDB persistent storage
            embedding_model: Sentence-transformers model name (runs on CPU)
        """
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model

        # Load embedding model on CPU to avoid GPU memory competition
        print(f"🔧 [RAG] Loading embedding model: {embedding_model} (CPU)...")
        self.embedder = SentenceTransformer(embedding_model, device="cpu")

        # Initialize ChromaDB with persistent storage
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="investigations",
            metadata={"hnsw:space": "cosine"},
        )

        print(f"✅ [RAG] Knowledge base ready: {self.count()} past cases")

    def add_investigation(self, history: List[str], verdict: dict,
                          investigation_id: str,
                          metadata: Optional[dict] = None):
        """Store a completed investigation for future retrieval.

        Args:
            history: List of timestamped observation strings
            verdict: Decision dict with status, confidence, reason
            investigation_id: Unique ID (e.g. "inv_00:01:23_1709900000")
            metadata: Optional extra metadata (timestamp, rounds, etc.)
        """
        # Combine history into a single text for embedding
        text = " ".join(history)
        if not text.strip():
            return

        # Avoid duplicate IDs
        existing = self.collection.get(ids=[investigation_id])
        if existing and existing["ids"]:
            return

        embedding = self.embedder.encode(text).tolist()

        doc_metadata = {
            "status": str(verdict.get("status", "UNKNOWN")),
            "confidence": int(verdict.get("confidence", 0)),
            "reason": str(verdict.get("reason", ""))[:500],  # Truncate for ChromaDB
        }
        if metadata:
            doc_metadata["timestamp"] = str(metadata.get("timestamp", ""))
            doc_metadata["rounds"] = int(metadata.get("rounds", 0))

        self.collection.add(
            ids=[investigation_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[doc_metadata],
        )

    def retrieve_similar(self, query_text: str, n_results: int = 3,
                         min_confidence: int = 0) -> List[dict]:
        """Find similar past investigations.

        Args:
            query_text: Text describing the current scene/observation
            n_results: Maximum number of cases to return
            min_confidence: Minimum confidence filter (0 = no filter)

        Returns:
            List of case dicts with: history, metadata, similarity
        """
        if self.count() == 0:
            return []

        embedding = self.embedder.encode(query_text).tolist()

        where_filter = None
        if min_confidence > 0:
            where_filter = {"confidence": {"$gte": min_confidence}}

        # Ensure n_results doesn't exceed collection size
        n_results = min(n_results, self.count())

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where_filter,
        )

        cases = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i] if results["distances"] else 0
                cases.append({
                    "id": results["ids"][0][i],
                    "history": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": round(1 - distance, 4),  # cosine distance → similarity
                })

        return cases

    def ingest_existing_alerts(self, alerts_dir: Path) -> int:
        """Bulk-import existing alerts from the alerts/ directory.

        Reads metadata.json from each alert subdirectory and adds the
        investigation to the knowledge base.

        Args:
            alerts_dir: Path to the alerts directory

        Returns:
            Number of investigations ingested
        """
        alerts_dir = Path(alerts_dir)
        if not alerts_dir.exists():
            print(f"⚠️  [RAG] Alerts directory not found: {alerts_dir}")
            return 0

        ingested = 0
        for alert_dir in sorted(alerts_dir.iterdir()):
            if not alert_dir.is_dir():
                continue

            metadata_path = alert_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path) as f:
                    meta = json.load(f)

                history = meta.get("history", [])
                verdict = meta.get("verdict", {})

                if not history:
                    continue

                self.add_investigation(
                    history=history,
                    verdict=verdict,
                    investigation_id=alert_dir.name,
                    metadata={
                        "timestamp": meta.get("timestamp", ""),
                        "rounds": len(history),
                    },
                )
                ingested += 1

            except Exception as e:
                print(f"   ⚠️  [RAG] Failed to ingest {alert_dir.name}: {e}")

        print(f"✅ [RAG] Ingested {ingested} investigations from {alerts_dir}")
        return ingested

    def count(self) -> int:
        """Return the number of investigations in the knowledge base."""
        return self.collection.count()

    def clear(self):
        """Delete all investigations from the knowledge base."""
        self.client.delete_collection("investigations")
        self.collection = self.client.get_or_create_collection(
            name="investigations",
            metadata={"hnsw:space": "cosine"},
        )
