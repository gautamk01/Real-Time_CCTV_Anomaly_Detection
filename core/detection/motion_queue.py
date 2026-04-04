"""Thread-safe fair scheduler for cross-camera motion events."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class MotionEvent:
    """Frozen trigger snapshot for a camera motion event."""

    camera_id: str
    timestamp: str
    trigger_frame_bgr: np.ndarray
    enqueued_at: float
    queue_depth: int


class MotionQueue:
    """
    Fair motion-event scheduler across cameras.

    Guarantees:
    - At most one pending event per camera.
    - A camera already under investigation is not re-queued.
    - Pending events for the same camera are refreshed with the newest snapshot.
    - Cameras are served round-robin in arrival order rather than newest-first.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._pending: Dict[str, MotionEvent] = {}
        self._pending_order: deque[str] = deque()
        self._inflight: set[str] = set()

    def put(self, camera_id: str, timestamp: str, frame_bgr: np.ndarray) -> bool:
        """
        Queue or refresh a motion event for a camera.

        Returns:
            True if the pending slot was created or refreshed.
            False if the camera is already inflight and the event was ignored.
        """
        event = MotionEvent(
            camera_id=camera_id,
            timestamp=timestamp,
            trigger_frame_bgr=frame_bgr.copy(),
            enqueued_at=time.time(),
            queue_depth=0,
        )

        with self._condition:
            if camera_id in self._inflight:
                return False

            if camera_id in self._pending:
                event.queue_depth = len(self._pending_order)
                self._pending[camera_id] = event
            else:
                self._pending[camera_id] = event
                self._pending_order.append(camera_id)
                event.queue_depth = len(self._pending_order)

            self._condition.notify()
            return True

    def wait_for_event(self, timeout: float = 0.1) -> Optional[MotionEvent]:
        """Block until the next fair-scheduled event is available."""
        deadline = time.time() + timeout

        with self._condition:
            while not self._pending_order:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)

            camera_id = self._pending_order.popleft()
            event = self._pending.pop(camera_id, None)
            if event is None:
                return None

            self._inflight.add(camera_id)
            return event

    def mark_done(self, camera_id: str) -> None:
        """Release a camera after its investigation finishes."""
        with self._condition:
            self._inflight.discard(camera_id)
            self._condition.notify_all()

    def clear_camera(self, camera_id: str) -> None:
        """Drop any pending or inflight state for a camera."""
        with self._condition:
            self._pending.pop(camera_id, None)
            self._inflight.discard(camera_id)
            self._pending_order = deque(
                queued_camera
                for queued_camera in self._pending_order
                if queued_camera != camera_id
            )
            self._condition.notify_all()

    def pending_count(self) -> int:
        """Return the number of cameras currently waiting for investigation."""
        with self._condition:
            return len(self._pending_order)
