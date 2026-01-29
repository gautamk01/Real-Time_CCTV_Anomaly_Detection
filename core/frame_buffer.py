"""Thread-safe Frame Buffer for Real-Time AI Access."""

import threading
from typing import Optional, Tuple
from PIL import Image


class FrameBuffer:
    """
    Thread-safe shared buffer for real-time frame access.
    
    Main thread continuously updates the latest frame.
    AI thread can grab the CURRENT frame whenever needed.
    """
    
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame: Optional[Image.Image] = None
        self.latest_timestamp: Optional[str] = None
        self.motion_detected = False
        self.has_new_motion = threading.Event()
    
    def update_frame(self, pil_image: Image.Image, timestamp: str, motion: bool = False):
        """
        Called by main thread to update the latest frame.
        
        Args:
            pil_image: PIL Image in RGB format
            timestamp: String timestamp (HH:MM:SS)
            motion: Whether motion was detected in this frame
        """
        with self.lock:
            self.latest_frame = pil_image
            self.latest_timestamp = timestamp
            if motion and not self.motion_detected:
                self.motion_detected = True
                self.has_new_motion.set()
    
    def get_current_frame(self) -> Tuple[Optional[Image.Image], Optional[str]]:
        """
        Called by AI thread to get the CURRENT frame.
        
        Returns:
            Tuple of (PIL Image, timestamp) or (None, None)
        """
        with self.lock:
            return self.latest_frame, self.latest_timestamp
    
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
