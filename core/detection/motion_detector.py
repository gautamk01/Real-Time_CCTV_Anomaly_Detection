"""OpenCV-based Motion Detection with Multi-Frame Gap."""

import cv2
import numpy as np
from typing import Tuple, Optional


class MotionDetector:
    """
    Real-time motion detection using multi-frame gap differencing.

    Instead of comparing consecutive frames (which misses slow/gradual motion
    at high FPS), this compares the current frame against a reference frame
    from N frames ago. It also uses accumulated contour area (total motion
    across all detected regions) rather than single-contour checking.
    """

    def __init__(self, threshold: int = 2000, blur_kernel: Tuple[int, int] = (5, 5),
                 binary_threshold: int = 20, dilate_iterations: int = 3,
                 skip_frames: int = 1, frame_gap: int = 5):
        """
        Initialize motion detector.

        Args:
            threshold: Minimum total contour area to consider as motion
            blur_kernel: Gaussian blur kernel size
            binary_threshold: Binary threshold value
            dilate_iterations: Number of dilation iterations
            skip_frames: Process every Nth frame (1 = every frame, 2 = every other)
            frame_gap: Compare current frame against N frames ago (higher = catches slower motion)
        """
        self.threshold = threshold
        self.blur_kernel = blur_kernel
        self.binary_threshold = binary_threshold
        self.dilate_iterations = dilate_iterations
        self.skip_frames = max(1, skip_frames)
        self.frame_gap = max(1, frame_gap)

        # Ring buffer to store recent grayscale frames for gap comparison
        self._gray_buffer = [None] * self.frame_gap
        self._buffer_idx = 0
        self._frame_counter = 0

    def detect(self, current_frame: np.ndarray) -> bool:
        """
        Detect motion by comparing current frame against a frame from N frames ago.

        Uses accumulated contour area (sum of all contour areas) instead of
        single-contour check, so distributed motion (e.g. a scuffle) is caught.

        Args:
            current_frame: BGR frame from OpenCV

        Returns:
            True if motion detected, False otherwise
        """
        self._frame_counter += 1

        # Skip frames if configured
        if self.skip_frames > 1 and (self._frame_counter % self.skip_frames) != 0:
            return False

        # Convert to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Get the reference frame from N frames ago
        ref_gray = self._gray_buffer[self._buffer_idx]

        # Store current frame in ring buffer (overwrites the oldest)
        self._gray_buffer[self._buffer_idx] = current_gray
        self._buffer_idx = (self._buffer_idx + 1) % self.frame_gap

        # Need at least frame_gap frames before we can compare
        if ref_gray is None:
            return False

        # Diff against the reference frame (N frames ago)
        diff = cv2.absdiff(ref_gray, current_gray)
        blur = cv2.GaussianBlur(diff, self.blur_kernel, 0)
        _, thresh = cv2.threshold(blur, self.binary_threshold, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=self.dilate_iterations)

        # Find contours and accumulate total motion area
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_area = sum(cv2.contourArea(c) for c in contours)

        return total_area > self.threshold

    def reset(self):
        """Reset the detector state."""
        self._gray_buffer = [None] * self.frame_gap
        self._buffer_idx = 0
        self._frame_counter = 0
