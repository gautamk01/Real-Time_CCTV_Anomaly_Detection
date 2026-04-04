"""Multi-Camera Manager — per-camera capture threads sharing one VLM."""

import cv2
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .frame_buffer import FrameBuffer
from .motion_detector import MotionDetector
from .motion_queue import MotionQueue


@dataclass
class CameraSource:
    """State for a single camera feed. Thread-safe for cross-thread access."""
    camera_id: str
    video_path: str
    frame_buffer: FrameBuffer
    motion_detector: MotionDetector
    fps: float = 30.0
    cooldown_seconds: float = 2.0
    min_consecutive_motion_frames: int = 2
    _is_active: bool = field(default=True, repr=False)
    _cooldown_until: float = field(default=0.0, repr=False)
    _motion_streak: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._is_active

    def deactivate(self) -> None:
        with self._lock:
            self._is_active = False

    def set_cooldown(self, until: float) -> None:
        with self._lock:
            self._cooldown_until = until
            self._motion_streak = 0

    def is_in_cooldown(self) -> bool:
        with self._lock:
            return time.time() < self._cooldown_until

    def update_motion_streak(self, motion_detected: bool) -> bool:
        """Track consecutive motion frames and return the gated result."""
        with self._lock:
            if motion_detected:
                self._motion_streak += 1
            else:
                self._motion_streak = 0
            return self._motion_streak >= self.min_consecutive_motion_frames


def _camera_capture_loop(
    cam: CameraSource,
    motion_queue: MotionQueue,
    stop_event: threading.Event,
    models_ready: threading.Event,
) -> None:
    """
    Capture loop for a single camera. Runs in its own thread.

    Reads frames, detects motion, updates per-camera FrameBuffer,
    and puts motion events into the shared MotionQueue.
    """
    # Wait for AI models before starting
    models_ready.wait()

    cap = cv2.VideoCapture(cam.video_path)
    if not cap.isOpened():
        print(f"   [CAM {cam.camera_id}] Failed to open: {cam.video_path}")
        cam.deactivate()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0
    cam.fps = fps
    frame_delay = 1.0 / fps

    print(f"   [CAM {cam.camera_id}] Capturing at {fps:.1f} FPS: {Path(cam.video_path).name}")

    frame_count = 0

    while cap.isOpened() and not stop_event.is_set():
        loop_start = time.time()

        ret, frame_bgr = cap.read()
        if not ret:
            break

        secs = frame_count / fps
        timestamp = time.strftime("%H:%M:%S", time.gmtime(secs))

        raw_motion = cam.motion_detector.detect(frame_bgr)
        gated_motion = cam.update_motion_streak(raw_motion)
        cam.frame_buffer.update_frame(frame_bgr, timestamp, motion=gated_motion)

        # Queue a trigger snapshot only after the motion gate opens.
        if gated_motion and not cam.is_in_cooldown():
            motion_queue.put(cam.camera_id, timestamp, frame_bgr)

        frame_count += 1

        # Real-time sync
        processing_time = time.time() - loop_start
        sleep_needed = frame_delay - processing_time
        if sleep_needed > 0:
            time.sleep(sleep_needed)

    cap.release()
    cam.deactivate()
    motion_queue.clear_camera(cam.camera_id)
    print(f"   [CAM {cam.camera_id}] Feed ended.")


class CameraManager:
    """
    Manages multiple camera capture threads with a shared MotionQueue.

    Each camera gets its own FrameBuffer and MotionDetector.
    Motion events from all cameras feed into one fair scheduler
    consumed by the single AI worker thread.
    """

    def __init__(
        self,
        camera_configs: Dict[str, str],
        motion_threshold: int = 2000,
        camera_cooldown: float = 2.0,
        min_consecutive_motion_frames: int = 2,
        camera_overrides: Optional[Dict[str, Dict[str, float]]] = None,
        blur_kernel: tuple = (5, 5),
        binary_threshold: int = 20,
        dilate_iterations: int = 3,
    ):
        """
        Args:
            camera_configs: {camera_id: video_path} mapping
            motion_threshold: MotionDetector threshold per camera
            camera_cooldown: Per-camera cooldown after each investigation
            min_consecutive_motion_frames: Frames of motion required to queue
            camera_overrides: Optional per-camera overrides for threshold/cooldown/gating
            blur_kernel: MotionDetector blur kernel
            binary_threshold: MotionDetector binary threshold
            dilate_iterations: MotionDetector dilation iterations
        """
        self.cameras: Dict[str, CameraSource] = {}
        self._threads: Dict[str, threading.Thread] = {}
        camera_overrides = camera_overrides or {}
        for cam_id, video_path in camera_configs.items():
            runtime_cfg = camera_overrides.get(cam_id, {})
            threshold = int(runtime_cfg.get("motion_threshold", motion_threshold))
            cooldown_seconds = float(
                runtime_cfg.get("cooldown_seconds", camera_cooldown)
            )
            min_motion_frames = int(
                runtime_cfg.get(
                    "min_consecutive_motion_frames",
                    min_consecutive_motion_frames,
                )
            )
            self.cameras[cam_id] = CameraSource(
                camera_id=cam_id,
                video_path=video_path,
                frame_buffer=FrameBuffer(),
                motion_detector=MotionDetector(
                    threshold=threshold,
                    blur_kernel=blur_kernel,
                    binary_threshold=binary_threshold,
                    dilate_iterations=dilate_iterations,
                ),
                cooldown_seconds=cooldown_seconds,
                min_consecutive_motion_frames=min_motion_frames,
            )

    def start_all(
        self,
        motion_queue: MotionQueue,
        stop_event: threading.Event,
        models_ready: threading.Event,
    ) -> None:
        """Spawn a capture thread for each camera."""
        for cam in self.cameras.values():
            thread = threading.Thread(
                target=_camera_capture_loop,
                args=(cam, motion_queue, stop_event, models_ready),
                daemon=True,
                name=f"cam-{cam.camera_id}",
            )
            self._threads[cam.camera_id] = thread
            thread.start()

    def stop_all(self) -> None:
        """Mark all cameras inactive (threads check stop_event independently)."""
        for cam in self.cameras.values():
            cam.deactivate()

    def get_camera(self, camera_id: str) -> Optional[CameraSource]:
        """Get camera by ID."""
        return self.cameras.get(camera_id)

    def any_active(self) -> bool:
        """Check if any camera is still active."""
        return any(cam.is_active for cam in self.cameras.values())

    def get_all_bgr_frames(self) -> Dict[str, Optional[np.ndarray]]:
        """Get latest BGR frame from each camera for display."""
        return {
            cam_id: cam.frame_buffer.get_latest_bgr()
            for cam_id, cam in self.cameras.items()
        }
