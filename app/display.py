"""Display helpers for single-camera and multi-camera views."""

import math
from pathlib import Path
import time
from typing import Dict, Optional

import cv2
import numpy as np

from core import CameraManager
from .runtime import AppRuntime


def make_display_grid(
    frames: Dict[str, Optional[np.ndarray]],
    target_height: int = 360,
) -> Optional[np.ndarray]:
    """Arrange camera frames into a grid for display."""

    valid = []
    for cam_id, frame in frames.items():
        if frame is None:
            continue
        height, width = frame.shape[:2]
        scale = target_height / height
        resized = cv2.resize(frame, (int(width * scale), target_height))
        cv2.putText(
            resized,
            cam_id,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        valid.append(resized)

    if not valid:
        return None

    if len(valid) == 1:
        return valid[0]

    cols = 2
    rows = math.ceil(len(valid) / cols)
    max_width = max(frame.shape[1] for frame in valid)
    padded = []
    for frame in valid:
        if frame.shape[1] < max_width:
            pad = np.zeros(
                (target_height, max_width - frame.shape[1], 3),
                dtype=np.uint8,
            )
            frame = np.hstack([frame, pad])
        padded.append(frame)

    while len(padded) < rows * cols:
        padded.append(np.zeros((target_height, max_width, 3), dtype=np.uint8))

    grid_rows = []
    for row in range(rows):
        grid_rows.append(np.hstack(padded[row * cols : (row + 1) * cols]))

    return np.vstack(grid_rows)


def display_loop(camera_manager: CameraManager, runtime: AppRuntime) -> None:
    """Display all camera feeds until the user quits or feeds end."""

    print("\n[DISPLAY] Waiting for AI models to load...")
    runtime.models_ready.wait()
    print("[DISPLAY] AI models ready, starting video display\n")

    cameras = camera_manager.cameras
    is_single = len(cameras) == 1
    camera_ids = list(cameras.keys())

    if is_single:
        window_name = f"Violence Detection - {camera_ids[0]}"
    else:
        window_name = f"Violence Detection - {len(cameras)} Cameras"

    print(f"[DISPLAY] Monitoring {len(cameras)} camera(s)")
    for cam_id, cam in cameras.items():
        print(f"   [{cam_id}] {Path(cam.video_path).name}")
    print("=" * 60)

    last_progress = time.time()

    while not runtime.stop_event.is_set():
        if not camera_manager.any_active():
            break

        if is_single:
            cam = cameras[camera_ids[0]]
            display_frame = cam.frame_buffer.get_latest_bgr()
            if display_frame is not None:
                timestamp, motion = cam.frame_buffer.get_status()
                cv2.putText(
                    display_frame,
                    timestamp or "",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                if motion:
                    cv2.putText(
                        display_frame,
                        "MOTION DETECTED",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                cv2.imshow(window_name, display_frame)
        else:
            all_frames = camera_manager.get_all_bgr_frames()
            for cam_id, frame in all_frames.items():
                if frame is None:
                    continue
                cam = cameras[cam_id]
                timestamp, motion = cam.frame_buffer.get_status()
                cv2.putText(
                    frame,
                    timestamp or "",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                if motion:
                    cv2.putText(
                        frame,
                        "MOTION",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
            grid = make_display_grid(all_frames)
            if grid is not None:
                cv2.imshow(window_name, grid)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[USER] Quit requested")
            runtime.stop_event.set()
            break

        now = time.time()
        if now - last_progress >= 2.0:
            active = sum(1 for camera in cameras.values() if camera.is_active)
            print(f"   ... [DISPLAY] {active}/{len(cameras)} cameras active ...")
            last_progress = now

    cv2.destroyAllWindows()
    runtime.stop_event.set()
    print("\n[DISPLAY] Video ended.")
