"""A2A Client — HTTP client for communicating with Edge, Cloud, and RAG agents.

Used by the investigator to send A2A messages to agent services over HTTP.
Handles base64 image encoding/decoding transparently so callers work with
PIL Images directly.
"""

import base64
import io
import time
from typing import Dict, List, Optional, Tuple

import httpx
from PIL import Image

from .protocol import AgentMessage, AgentResponse


class A2AClient:
    """HTTP client for A2A agent communication.

    Provides high-level methods matching the investigator's needs:
    observe, encode, answer (edge), assess (cloud), retrieve/ingest (RAG).
    """

    def __init__(self, edge_url: str, cloud_url: str,
                 rag_url: Optional[str] = None,
                 timeout: float = 15.0):
        """
        Args:
            edge_url: Edge agent base URL (e.g. "http://localhost:8001")
            cloud_url: Cloud agent base URL (e.g. "http://localhost:8002")
            rag_url: RAG agent base URL (e.g. "http://localhost:8003"), or None
            timeout: HTTP request timeout in seconds
        """
        self.edge_url = edge_url.rstrip("/")
        self.cloud_url = cloud_url.rstrip("/")
        self.rag_url = rag_url.rstrip("/") if rag_url else None
        self.http = httpx.Client(timeout=timeout)
        self._sender = "investigator"

    # --- Image helpers ---

    @staticmethod
    def _image_to_b64(image: Image.Image) -> str:
        """Encode PIL Image to base64 JPEG string."""
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # --- Internal transport ---

    def _send(self, url: str, msg_type: str, payload: dict,
              correlation_id: str = "",
              max_retries: int = 3, backoff: float = 1.0) -> AgentResponse:
        """Send an AgentMessage and parse the AgentResponse.

        Retries on connection errors with exponential backoff to handle
        agent startup delays gracefully.
        """
        msg = AgentMessage(
            msg_type=msg_type,
            sender=self._sender,
            payload=payload,
            correlation_id=correlation_id,
        )
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.http.post(url, json=msg.model_dump())
                resp.raise_for_status()
                return AgentResponse(**resp.json())
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                last_exc = e
                if attempt < max_retries:
                    wait = backoff * (2 ** (attempt - 1))
                    print(f"   ⚠️  [A2A] Connection to {url} failed (attempt {attempt}/{max_retries}), retrying in {wait:.0f}s...")
                    time.sleep(wait)
        raise RuntimeError(
            f"[A2A] Could not reach {url} after {max_retries} attempts: {last_exc}"
        )

    # --- Edge Agent Methods ---

    def observe(
        self,
        image: Image.Image,
        question: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Full observation: encode + answer in one call.

        Args:
            image: PIL Image (RGB)
            question: Question to ask about the image

        Returns:
            Description string from the vision model
        """
        payload = {"image_b64": self._image_to_b64(image), "question": question}
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        response = self._send(
            f"{self.edge_url}/observe",
            msg_type="observe",
            payload=payload,
        )
        if response.status != "success":
            raise RuntimeError(f"Edge observe failed: {response.error_detail}")
        return response.payload["description"]

    def encode(self, image: Image.Image) -> str:
        """Phase 1: Encode image into embeddings (heavy GPU work).

        Args:
            image: PIL Image (RGB)

        Returns:
            encode_id — reference for use with answer()
        """
        response = self._send(
            f"{self.edge_url}/encode",
            msg_type="encode",
            payload={"image_b64": self._image_to_b64(image)},
        )
        if response.status != "success":
            raise RuntimeError(f"Edge encode failed: {response.error_detail}")
        return response.payload["encode_id"]

    def answer(
        self,
        encode_id: str,
        question: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Phase 2: Answer question using pre-encoded image (lightweight).

        Args:
            encode_id: Reference from a previous encode() call
            question: Question to ask

        Returns:
            Answer string from the vision model
        """
        payload = {"encode_id": encode_id, "question": question}
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        response = self._send(
            f"{self.edge_url}/answer",
            msg_type="answer",
            payload=payload,
        )
        if response.status != "success":
            raise RuntimeError(f"Edge answer failed: {response.error_detail}")
        return response.payload["description"]

    # --- Cloud Agent Methods ---

    def assess(self, history: List[str],
               rag_context: Optional[List[dict]] = None) -> Dict:
        """Assess threat level from observation history.

        Args:
            history: List of timestamped observation strings
            rag_context: Optional list of similar past cases from RAG

        Returns:
            Decision dict with: status, confidence, question, reason
        """
        payload = {"history": history}
        if rag_context is not None:
            payload["rag_context"] = rag_context

        response = self._send(
            f"{self.cloud_url}/assess",
            msg_type="assess",
            payload=payload,
        )
        # Return the payload even on error — cloud_agent.py provides
        # a safe fallback decision dict in its error response
        return response.payload

    # --- RAG Agent Methods ---

    @property
    def has_rag(self) -> bool:
        """Whether a RAG agent URL is configured."""
        return self.rag_url is not None

    def retrieve(self, query_text: str, limit: int = 3,
                 min_confidence: int = 0) -> List[dict]:
        """Retrieve similar past investigations from RAG.

        Args:
            query_text: Text to find similar cases for
            limit: Max number of cases to return
            min_confidence: Minimum confidence filter

        Returns:
            List of case dicts with: history, metadata, similarity
        """
        if not self.rag_url:
            return []

        try:
            response = self._send(
                f"{self.rag_url}/retrieve",
                msg_type="retrieve",
                payload={
                    "query_text": query_text,
                    "limit": limit,
                    "min_confidence": min_confidence,
                },
            )
            if response.status != "success":
                return []
            return response.payload.get("cases", [])
        except Exception as e:
            print(f"   ⚠️  [A2A] RAG retrieve failed: {e}")
            return []

    def ingest(self, history: List[str], verdict: dict,
               investigation_id: str, metadata: Optional[dict] = None):
        """Ingest a completed investigation into the RAG knowledge base.

        Args:
            history: Investigation observation history
            verdict: Final decision dict
            investigation_id: Unique ID for this investigation
            metadata: Optional extra metadata
        """
        if not self.rag_url:
            return

        try:
            self._send(
                f"{self.rag_url}/ingest",
                msg_type="ingest",
                payload={
                    "history": history,
                    "verdict": verdict,
                    "investigation_id": investigation_id,
                    "metadata": metadata or {},
                },
            )
        except Exception as e:
            print(f"   ⚠️  [A2A] RAG ingest failed: {e}")

    # --- Discovery ---

    def get_agent_card(self, agent: str) -> dict:
        """Fetch an agent's capability card.

        Args:
            agent: "edge", "cloud", or "rag"
        """
        urls = {"edge": self.edge_url, "cloud": self.cloud_url, "rag": self.rag_url}
        url = urls.get(agent)
        if not url:
            raise ValueError(f"Unknown agent: {agent}")
        resp = self.http.get(f"{url}/agent-card")
        resp.raise_for_status()
        return resp.json()

    def wait_for_agents(self, timeout: float = 30.0, poll_interval: float = 2.0):
        """Block until all configured agents are reachable.

        Args:
            timeout: Max seconds to wait before giving up
            poll_interval: Seconds between health check attempts

        Raises:
            RuntimeError: If agents are not reachable within timeout
        """
        agents = {"Edge": self.edge_url, "Cloud": self.cloud_url}
        if self.rag_url:
            agents["RAG"] = self.rag_url

        start = time.time()
        pending = dict(agents)

        while pending and (time.time() - start) < timeout:
            for name, url in list(pending.items()):
                try:
                    resp = self.http.get(f"{url}/agent-card", timeout=3.0)
                    if resp.status_code == 200:
                        print(f"   ✅ {name} agent ready at {url}")
                        del pending[name]
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    pass
            if pending:
                remaining = list(pending.keys())
                elapsed = time.time() - start
                print(f"   ⏳ Waiting for agents: {', '.join(remaining)} ({elapsed:.0f}s / {timeout:.0f}s)")
                time.sleep(poll_interval)

        if pending:
            raise RuntimeError(
                f"[A2A] Agents not reachable after {timeout}s: {', '.join(pending.keys())}"
            )
        print(f"   ✅ All agents ready!")

    def close(self):
        """Close the HTTP client."""
        self.http.close()
