
from core import (
    FrameBuffer, MotionDetector, AIInvestigator, MetricsLogger,
    MotionQueue, CameraManager, InferenceServer, CameraPipeline,
)
from models import EdgeVision, CloudAI
from config import Config
import os
import sys
import cv2
import time
import math
import threading
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Type
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


# Global shared resources
stop_event = threading.Event()
models_ready = threading.Event()  # Signal when AI models are loaded
metrics_logger = MetricsLogger()


def _init_models(config: Type[Config]):
    """Initialize AI models and return (edge_vision, cloud_ai, a2a_client).

    Handles A2A discovery and local model fallback.
    """
    a2a_client = None
    if config.ENABLE_A2A:
        try:
            from a2a import A2AClient
            rag_url = config.RAG_AGENT_URL if config.ENABLE_RAG else None
            a2a_client = A2AClient(
                edge_url=config.EDGE_AGENT_URL,
                cloud_url=config.CLOUD_AGENT_URL,
                rag_url=rag_url,
            )
            print(f"[A2A] Networked agent communication enabled")
            print(f"   Edge:  {config.EDGE_AGENT_URL}")
            print(f"   Cloud: {config.CLOUD_AGENT_URL}")
            if rag_url:
                print(f"   RAG:   {config.RAG_AGENT_URL}")

            print(f"\n[A2A] Waiting for agent services to be ready...")
            try:
                a2a_client.wait_for_agents(timeout=60.0)
            except RuntimeError as e:
                print(f"[A2A] {e}")
                print(f"   Falling back to direct model calls")
                a2a_client = None
        except Exception as e:
            print(f"[A2A] Could not initialize: {e}")
            print(f"   Falling back to direct model calls")

    edge_vision = None
    cloud_ai = None

    if a2a_client is not None:
        print("[AI] A2A mode — using remote agents, skipping local model loading")
    else:
        print("[AI] Loading local models...")
        print("[AI] Please wait, this may take a few minutes on first run...")
        edge_vision = EdgeVision(
            model_id=config.EDGE_MODEL_ID,
            device=config.DEVICE,
            quant_mode=config.EDGE_QUANT_MODE,
        )
        if config.CLOUD_BACKEND == "ollama":
            cloud_ai = CloudAI(
                api_key="ollama",
                model_id=config.CLOUD_MODEL_ID,
                base_url="http://localhost:11434/v1",
            )
        else:
            cloud_ai = CloudAI(
                api_key=config.GROQ_API_KEY,
                model_id=config.CLOUD_MODEL_ID,
            )

    return edge_vision, cloud_ai, a2a_client


def _init_fcm(config: Type[Config]):
    """Initialize FCM notifier if configured."""
    if not config.ENABLE_FCM_NOTIFICATIONS:
        return None
    try:
        from core.fcm_notifier import FCMNotifier
        return FCMNotifier(
            credentials_path=config.FIREBASE_CREDENTIALS_PATH,
            topic=config.FCM_TOPIC,
            enable_review_notifications=config.ENABLE_REVIEW_NOTIFICATIONS,
        )
    except Exception as e:
        print(f"[FCM] Could not initialize: {e}")
        print(f"   Continuing without mobile notifications")
        return None


def _warmup_gpu(inference_server) -> None:
    """Run one dummy forward pass to bring GPU to full operating speed.

    Without warm-up, the first real inference takes ~2.24s (quantization
    kernels are JIT-compiled on first call). Subsequent inferences take ~1.5s.
    Warming up before capture starts ensures every investigation gets full speed.

    MUST be called before models_ready.set() so capture threads don't start
    until the GPU is ready.
    """
    print("[GPU] Warming up GPU (dummy forward pass)...")
    try:
        dummy = Image.new("RGB", (64, 64), color=(128, 128, 128))
        future = inference_server.submit_analyze(
            dummy, "Describe this image.", max_tokens=8, priority=2, camera_id="warmup"
        )
        future.result(timeout=30.0)
        print("[GPU] Warm-up complete — GPU at full speed")
    except Exception as e:
        print(f"[GPU] Warm-up skipped ({e}) — first inference may be slower")


def distributed_ai_startup(
    config: Type[Config],
    camera_manager: CameraManager,
):
    """Initialize AI subsystem using distributed pipeline architecture.

    Creates:
    1. InferenceServer — single model, shared GPU queue
    2. CameraPipeline per camera — independent investigation threads

    Returns:
        (inference_server, pipelines, a2a_client)
    """
    print("\n[AI] Initializing distributed AI pipeline...")

    edge_vision, cloud_ai, a2a_client = _init_models(config)
    fcm_notifier = _init_fcm(config)

    effective_max_rounds = config.get_effective_max_rounds()
    if effective_max_rounds != config.MAX_INVESTIGATION_ROUNDS:
        print(
            "[AI] Multi-camera throughput mode enabled "
            f"(max rounds capped at {effective_max_rounds})"
        )

    # Create InferenceServer (wraps single EdgeVision in a thread-safe queue)
    inference_server = None
    if edge_vision is not None:
        inference_server = InferenceServer(
            edge_vision,
            max_queue_size=config.GPU_QUEUE_SIZE,
        )
        inference_server.start()
        _warmup_gpu(inference_server)

    # Create per-camera pipelines
    pipelines: Dict[str, CameraPipeline] = {}
    cameras = camera_manager.cameras

    for cam_id, cam in cameras.items():
        def make_frame_provider(cid=cam_id):
            def provider():
                c = camera_manager.get_camera(cid)
                if c is None:
                    return None, None
                return c.frame_buffer.get_current_frame()
            return provider

        pipeline = CameraPipeline(
            camera_id=cam_id,
            inference_server=inference_server,
            cloud_ai=cloud_ai,
            frame_provider=make_frame_provider(),
            max_rounds=effective_max_rounds,
            fps=cam.fps,
            initial_edge_max_tokens=config.EDGE_INITIAL_MAX_TOKENS,
            followup_edge_max_tokens=config.EDGE_FOLLOWUP_MAX_TOKENS,
            save_alerts=config.SAVE_ALERTS,
            alerts_dir=config.ALERTS_DIR,
            buffer_duration=config.BUFFER_DURATION_SECONDS,
            fcm_notifier=fcm_notifier,
            metrics_logger=metrics_logger,
            a2a_client=a2a_client,
            video_file=f"{cam_id}:{Path(cam.video_path).name}",
        )
        pipelines[cam_id] = pipeline

    return inference_server, pipelines, a2a_client


def distributed_camera_capture_loop(
    cam,
    pipelines: Dict[str, CameraPipeline],
    stop_event: threading.Event,
    models_ready: threading.Event,
) -> None:
    """Capture loop for distributed architecture.

    Instead of pushing events into a shared MotionQueue, each camera
    submits motion events directly to its own CameraPipeline.
    """
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

    pipeline = pipelines.get(cam.camera_id)
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

        # Submit motion events to per-camera pipeline (slot pattern)
        # Pipeline manages its own adaptive cooldown — no cam.is_in_cooldown() needed
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


def _make_display_grid(
    frames: Dict[str, Optional[np.ndarray]],
    target_height: int = 360,
) -> Optional[np.ndarray]:
    """Arrange camera frames into a grid for display."""
    valid = []
    for cam_id, frame in frames.items():
        if frame is None:
            continue
        h, w = frame.shape[:2]
        scale = target_height / h
        resized = cv2.resize(frame, (int(w * scale), target_height))
        cv2.putText(
            resized, cam_id, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        valid.append(resized)

    if not valid:
        return None

    if len(valid) == 1:
        return valid[0]

    cols = 2
    rows = math.ceil(len(valid) / cols)
    max_w = max(f.shape[1] for f in valid)
    padded = []
    for f in valid:
        if f.shape[1] < max_w:
            pad = np.zeros((target_height, max_w - f.shape[1], 3), dtype=np.uint8)
            f = np.hstack([f, pad])
        padded.append(f)
    while len(padded) < rows * cols:
        padded.append(np.zeros((target_height, max_w, 3), dtype=np.uint8))

    grid_rows = []
    for r in range(rows):
        grid_rows.append(np.hstack(padded[r * cols:(r + 1) * cols]))
    return np.vstack(grid_rows)


def display_loop(camera_manager: CameraManager) -> None:
    """Main thread: display all camera feeds in a grid."""
    print("\n[DISPLAY] Waiting for AI models to load...")
    models_ready.wait()
    print("[DISPLAY] AI models ready, starting video display\n")

    cameras = camera_manager.cameras
    is_single = len(cameras) == 1
    cam_ids = list(cameras.keys())

    if is_single:
        window_name = f'Violence Detection - {cam_ids[0]}'
    else:
        window_name = f'Violence Detection - {len(cameras)} Cameras'

    print(f"[DISPLAY] Monitoring {len(cameras)} camera(s)")
    for cam_id, cam in cameras.items():
        print(f"   [{cam_id}] {Path(cam.video_path).name}")
    print("=" * 60)

    last_progress = time.time()

    while not stop_event.is_set():
        if not camera_manager.any_active():
            break

        if is_single:
            cam = cameras[cam_ids[0]]
            display_frame = cam.frame_buffer.get_latest_bgr()
            if display_frame is not None:
                ts, motion = cam.frame_buffer.get_status()
                cv2.putText(
                    display_frame, ts or "", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )
                if motion:
                    cv2.putText(
                        display_frame, "MOTION DETECTED", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    )
                cv2.imshow(window_name, display_frame)
        else:
            all_frames = camera_manager.get_all_bgr_frames()
            for cam_id, frame in all_frames.items():
                if frame is not None:
                    cam = cameras[cam_id]
                    ts, motion = cam.frame_buffer.get_status()
                    cv2.putText(
                        frame, ts or "", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )
                    if motion:
                        cv2.putText(
                            frame, "MOTION", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        )
            grid = _make_display_grid(all_frames)
            if grid is not None:
                cv2.imshow(window_name, grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[USER] Quit requested")
            stop_event.set()
            break

        now = time.time()
        if now - last_progress >= 2.0:
            active = sum(1 for c in cameras.values() if c.is_active)
            print(f"   ... [DISPLAY] {active}/{len(cameras)} cameras active ...")
            last_progress = now

    cv2.destroyAllWindows()
    stop_event.set()
    print("\n[DISPLAY] Video ended.")


def main():
    """Main entry point — distributed multi-camera architecture."""
    print("\n" + "=" * 60)
    print("REAL-TIME VIOLENCE DETECTION SYSTEM")
    print("   Distributed GPU Inference Architecture")
    print("=" * 60)

    if not Config.validate():
        sys.exit(1)

    Config.print_info()

    # Build camera configs
    camera_configs = Config.get_cameras()

    # Create camera manager
    camera_manager = CameraManager(
        camera_configs=camera_configs,
        motion_threshold=Config.MOTION_THRESHOLD,
        camera_cooldown=Config.CAMERA_COOLDOWN,
        min_consecutive_motion_frames=Config.MIN_CONSECUTIVE_MOTION_FRAMES,
        camera_overrides=Config.get_camera_runtime_settings(),
        blur_kernel=Config.MOTION_BLUR_KERNEL,
        binary_threshold=Config.MOTION_BINARY_THRESHOLD,
        dilate_iterations=Config.MOTION_DILATE_ITERATIONS,
    )

    # Initialize distributed AI subsystem
    inference_server, pipelines, a2a_client = distributed_ai_startup(
        Config, camera_manager,
    )

    # Signal models are ready
    models_ready.set()

    # Start per-camera pipelines (investigation threads)
    for pipeline in pipelines.values():
        pipeline.start(stop_event)

    # Start camera capture threads (direct to per-camera pipelines)
    capture_threads = []
    for cam in camera_manager.cameras.values():
        t = threading.Thread(
            target=distributed_camera_capture_loop,
            args=(cam, pipelines, stop_event, models_ready),
            daemon=True,
            name=f"cam-{cam.camera_id}",
        )
        capture_threads.append(t)
        t.start()

    try:
        display_loop(camera_manager)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        print("\nShutting down...")
        stop_event.set()

        # Stop camera captures
        camera_manager.stop_all()
        for t in capture_threads:
            t.join(timeout=3)

        # Stop investigation pipelines
        for pipeline in pipelines.values():
            pipeline.stop(timeout=3)

        # Shutdown GPU inference server
        if inference_server:
            stats = inference_server.get_stats()
            inference_server.shutdown()
            print(f"\n[GPU SERVER] Final stats:")
            print(f"   Requests processed: {stats['requests_processed']}")
            print(f"   Encode calls: {stats['encode_calls']}")
            print(f"   Answer calls: {stats['answer_calls']}")
            print(f"   Errors: {stats['errors']}")
            print(f"   Total GPU time: {stats['total_gpu_time_ms']:.0f}ms")

        # Print benchmark metrics
        metrics_logger.print_summary()
        metrics_logger.export_json("alerts/benchmark_metrics.json")
        metrics_logger.export_csv("alerts/benchmark_metrics.csv")

        print("Shutdown complete\n")


if __name__ == "__main__":
    main()
