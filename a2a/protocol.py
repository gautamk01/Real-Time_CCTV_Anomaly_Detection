"""A2A (Agent-to-Agent) Protocol — Shared message types and agent interface.

Defines the standardized message format for networked communication between
Edge, Cloud, and RAG agents via FastAPI HTTP endpoints.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """Standardized request message sent between agents."""

    msg_id: str = Field(default_factory=lambda: str(uuid4()))
    msg_type: str  # "observe", "encode", "answer", "assess", "retrieve", "ingest"
    sender: str  # Agent name (e.g. "investigator", "edge_vision")
    payload: Dict[str, Any]
    timestamp: float = Field(default_factory=time.time)
    correlation_id: str = ""  # Links request/response pairs across agents


class AgentResponse(BaseModel):
    """Standardized response returned by an agent."""

    msg_id: str = Field(default_factory=lambda: str(uuid4()))
    status: str  # "success" or "error"
    payload: Dict[str, Any]
    processing_time_ms: float = 0.0
    correlation_id: str = ""
    error_detail: Optional[str] = None


class AgentCard(BaseModel):
    """Describes an agent's identity and capabilities for discovery."""

    name: str
    description: str
    capabilities: List[str]  # Message types this agent handles
    endpoints: List[Dict[str, str]]  # [{path, method, msg_type, description}]
    metadata: Dict[str, Any] = {}


class AgentInterface(ABC):
    """Abstract base class that all A2A agents must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent name."""
        ...

    @abstractmethod
    def get_agent_card(self) -> AgentCard:
        """Return this agent's capability card."""
        ...

    @abstractmethod
    def handle(self, message: AgentMessage) -> AgentResponse:
        """Process an incoming message and return a response."""
        ...
