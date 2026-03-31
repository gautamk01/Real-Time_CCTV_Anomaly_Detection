"""Thread-safe Frame Buffer for Real-Time AI Access.

Optimized: stores raw BGR numpy arrays and lazily converts to PIL
only when the AI thread actually requests a frame. Also provides
direct RGB numpy access to skip the PIL round-trip when possible.
"""

import cv2
import threading
import numpy as np
from typing import Optional, Tuple
from PIL import Image


class FrameBuffer:
    """
    Thread-safe shared buffer for real-time frame access.

    Main thread continuously updates the latest BGR frame.
    AI thread can grab the CURRENT frame as a PIL Image (lazy conversion)
    or as an RGB numpy array (avoids PIL overhead).
    """

    def __init__(self):
        self.lock = threading.Lock()
        self._latest_bgr: Optional[np.ndarray] = None
        self.latest_timestamp: Optional[str] = None
        self.motion_detected = False
        self.has_new_motion = threading.Event()

        # Lazy conversion caches (invalidated on each BGR update)
        self._pil_cache: Optional[Image.Image] = None
        self._pil_dirty = True
        self._rgb_cache: Optional[np.ndarray] = None
        self._rgb_dirty = True

    def update_frame(self, frame_bgr: np.ndarray, timestamp: str, motion: bool = False):
        """
        Called by main thread to update the latest frame.

        Args:
            frame_bgr: BGR numpy array from OpenCV
            timestamp: String timestamp (HH:MM:SS)
            motion: Whether motion was detected in this frame
        """
        with self.lock:
            self._latest_bgr = frame_bgr
            self.latest_timestamp = timestamp
            self._pil_dirty = True
            self._rgb_dirty = True
            if motion and not self.motion_detected:
                self.motion_detected = True
                self.has_new_motion.set()

    def get_current_frame(self) -> Tuple[Optional[Image.Image], Optional[str]]:
        """
        Called by AI thread to get the CURRENT frame as PIL Image.

        Lazily converts BGR→RGB→PIL only when called (not on every frame update).

        Returns:
            Tuple of (PIL Image, timestamp) or (None, None)
        """
        with self.lock:
            if self._latest_bgr is None:
                return None, None
            if self._pil_dirty or self._pil_cache is None:
                rgb = self._get_rgb_locked()
                self._pil_cache = Image.fromarray(rgb)
                self._pil_dirty = False
            return self._pil_cache, self.latest_timestamp

    def _get_rgb_locked(self) -> np.ndarray:
        """Convert BGR to RGB (must be called with lock held)."""
        if self._rgb_dirty or self._rgb_cache is None:
            self._rgb_cache = cv2.cvtColor(self._latest_bgr, cv2.COLOR_BGR2RGB)
            self._rgb_dirty = False
        return self._rgb_cache

    def get_latest_bgr(self) -> Optional[np.ndarray]:
        """Get raw BGR frame for display (thread-safe copy)."""
        with self.lock:
            return self._latest_bgr.copy() if self._latest_bgr is not None else None

    def get_status(self) -> Tuple[Optional[str], bool]:
        """Return (latest_timestamp, motion_detected) atomically."""
        with self.lock:
            return self.latest_timestamp, self.motion_detected

    def clear_motion_flag(self):
        """Called by AI thread after handling motion event."""
        with self.lock:
            self.motion_detected = False
            self.has_new_motion.clear()

    def wait_for_motion(self, timeout: float = 0.1) -> bool:
        """
        Wait for motion detection signal.

        Args:
            timeout: Timeout in seconds

        Returns:
            True if motion was detected, False if timeout
        """
        return self.has_new_motion.wait(timeout)
