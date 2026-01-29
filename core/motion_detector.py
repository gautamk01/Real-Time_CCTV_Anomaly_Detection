"""OpenCV-based Motion Detection."""

import cv2
import numpy as np
from typing import Tuple, Optional


class MotionDetector:
    """
    Real-time motion detection using frame differencing.
    """
    
    def __init__(self, threshold: int = 2000, blur_kernel: Tuple[int, int] = (5, 5),
                 binary_threshold: int = 20, dilate_iterations: int = 3):
        """
        Initialize motion detector.
        
        Args:
            threshold: Minimum contour area to consider as motion
            blur_kernel: Gaussian blur kernel size
            binary_threshold: Binary threshold value
            dilate_iterations: Number of dilation iterations
        """
        self.threshold = threshold
        self.blur_kernel = blur_kernel
        self.binary_threshold = binary_threshold
        self.dilate_iterations = dilate_iterations
        
        self.prev_frame: Optional[np.ndarray] = None
    
    def detect(self, current_frame: np.ndarray) -> bool:
        """
        Detect motion between previous and current frame.
        
        Args:
            current_frame: BGR frame from OpenCV
            
        Returns:
            True if motion detected, False otherwise
        """
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return False
        
        # Calculate frame difference
        diff = cv2.absdiff(self.prev_frame, current_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        _, thresh = cv2.threshold(blur, self.binary_threshold, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=self.dilate_iterations)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any contour exceeds threshold
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > self.threshold:
                motion_detected = True
                break
        
        # Update previous frame
        self.prev_frame = current_frame.copy()
        
        return motion_detected
    
    def reset(self):
        """Reset the detector state."""
        self.prev_frame = None
