"""High-level application orchestration for startup and shutdown."""

from pathlib import Path

from core import CameraManager

from .bootstrap import StartupArtifacts, build_startup_artifacts
from .capture import start_capture_threads
from .display import display_loop
from .runtime import AppRuntime


def build_camera_manager(config) -> CameraManager:
    """Create the camera manager from the config facade."""

    return CameraManager(
        camera_configs=config.get_cameras(),
        motion_threshold=config.MOTION_THRESHOLD,
        camera_cooldown=config.CAMERA_COOLDOWN,
        min_consecutive_motion_frames=config.MIN_CONSECUTIVE_MOTION_FRAMES,
        camera_overrides=config.get_camera_runtime_settings(),
        blur_kernel=config.MOTION_BLUR_KERNEL,
        binary_threshold=config.MOTION_BINARY_THRESHOLD,
        dilate_iterations=config.MOTION_DILATE_ITERATIONS,
    )


def _print_gpu_server_stats(inference_server) -> None:
    stats = inference_server.get_stats()
    inference_server.shutdown()
    print("\n[GPU SERVER] Final stats:")
    print(f"   Requests processed: {stats['requests_processed']}")
    print(f"   Encode calls: {stats['encode_calls']}")
    print(f"   Answer calls: {stats['answer_calls']}")
    print(f"   Errors: {stats['errors']}")
    print(f"   Total GPU time: {stats['total_gpu_time_ms']:.0f}ms")


def shutdown_application(
    config,
    runtime: AppRuntime,
    camera_manager: CameraManager,
    capture_threads,
    startup: StartupArtifacts,
) -> None:
    """Stop threads and export metrics for the current run."""

    print("\nShutting down...")
    runtime.stop_event.set()

    camera_manager.stop_all()
    for thread in capture_threads:
        thread.join(timeout=3)

    for pipeline in startup.pipelines.values():
        pipeline.stop(timeout=3)

    if startup.inference_server is not None:
        _print_gpu_server_stats(startup.inference_server)

    metrics_dir = Path(config.ALERTS_DIR)
    runtime.metrics_logger.print_summary()
    runtime.metrics_logger.export_json(str(metrics_dir / "benchmark_metrics.json"))
    runtime.metrics_logger.export_csv(str(metrics_dir / "benchmark_metrics.csv"))

    print("Shutdown complete\n")


def run_application(config) -> None:
    """Run the distributed multi-camera application lifecycle."""

    runtime = AppRuntime()
    camera_manager = build_camera_manager(config)
    startup = build_startup_artifacts(
        config,
        camera_manager,
        runtime.metrics_logger,
    )

    runtime.models_ready.set()

    for pipeline in startup.pipelines.values():
        pipeline.start(runtime.stop_event)

    capture_threads = start_capture_threads(camera_manager, startup.pipelines, runtime)

    try:
        display_loop(camera_manager, runtime)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        shutdown_application(
            config,
            runtime,
            camera_manager,
            capture_threads,
            startup,
        )
