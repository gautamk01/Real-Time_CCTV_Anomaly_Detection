"""Shared runtime state for the application entrypoint."""

from dataclasses import dataclass, field
import threading

from core import MetricsLogger


@dataclass
class AppRuntime:
    """Process-wide runtime state shared across app modules."""

    stop_event: threading.Event = field(default_factory=threading.Event)
    models_ready: threading.Event = field(default_factory=threading.Event)
    metrics_logger: MetricsLogger = field(default_factory=MetricsLogger)
