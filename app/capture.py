"""Camera capture loop helpers."""

from pathlib import Path
import threading
import time
from typing import Dict, List

import cv2

from core import CameraManager, CameraPipeline
from .runtime import AppRuntime


def camera_capture_loop(
    cam,
    pipelines: Dict[str, CameraPipeline],
    runtime: AppRuntime,
) -> None:
    """Read frames from one camera and hand motion events to its pipeline."""

    runtime.models_ready.wait()

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

    pipeline = pipelines.get(cam.camera_id)
    print(
        f"   [CAM {cam.camera_id}] Capturing at {fps:.1f} FPS: "
        f"{Path(cam.video_path).name}"
    )

    frame_count = 0

    while cap.isOpened() and not runtime.stop_event.is_set():
        loop_start = time.time()

        ret, frame_bgr = cap.read()
        if not ret:
            break

        secs = frame_count / fps
        timestamp = time.strftime("%H:%M:%S", time.gmtime(secs))

        raw_motion = cam.motion_detector.detect(frame_bgr)
        gated_motion = cam.update_motion_streak(raw_motion)
        cam.frame_buffer.update_frame(frame_bgr, timestamp, motion=gated_motion)

        if gated_motion and pipeline:
            pipeline.submit_motion(timestamp, frame_bgr)

        frame_count += 1

        processing_time = time.time() - loop_start
        sleep_needed = frame_delay - processing_time
        if sleep_needed > 0:
            time.sleep(sleep_needed)

    cap.release()
    cam.deactivate()
    print(f"   [CAM {cam.camera_id}] Feed ended.")


def start_capture_threads(
    camera_manager: CameraManager,
    pipelines: Dict[str, CameraPipeline],
    runtime: AppRuntime,
) -> List[threading.Thread]:
    """Spawn one capture thread per configured camera."""

    capture_threads = []
    for cam in camera_manager.cameras.values():
        thread = threading.Thread(
            target=camera_capture_loop,
            args=(cam, pipelines, runtime),
            daemon=True,
            name=f"cam-{cam.camera_id}",
        )
        capture_threads.append(thread)
        thread.start()

    return capture_threads
