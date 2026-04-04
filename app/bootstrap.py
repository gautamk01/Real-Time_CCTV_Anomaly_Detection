"""Startup helpers for model loading and pipeline assembly."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from core import CameraManager, CameraPipeline, InferenceServer


@dataclass
class StartupArtifacts:
    """Application services created during startup."""

    inference_server: Optional[InferenceServer]
    pipelines: Dict[str, CameraPipeline]
    a2a_client: Any = None


def _init_models(config):
    """Initialize AI models and return (edge_vision, cloud_ai, a2a_client)."""

    from models import CloudAI, EdgeVision

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
            print("[A2A] Networked agent communication enabled")
            print(f"   Edge:  {config.EDGE_AGENT_URL}")
            print(f"   Cloud: {config.CLOUD_AGENT_URL}")
            if rag_url:
                print(f"   RAG:   {config.RAG_AGENT_URL}")

            print("\n[A2A] Waiting for agent services to be ready...")
            try:
                a2a_client.wait_for_agents(timeout=60.0)
            except RuntimeError as exc:
                print(f"[A2A] {exc}")
                print("   Falling back to direct model calls")
                a2a_client = None
        except Exception as exc:
            print(f"[A2A] Could not initialize: {exc}")
            print("   Falling back to direct model calls")

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


def _init_fcm(config):
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
    except Exception as exc:
        print(f"[FCM] Could not initialize: {exc}")
        print("   Continuing without mobile notifications")
        return None


def _warmup_gpu(inference_server: InferenceServer) -> None:
    """Run one dummy forward pass so the first real inference is not cold."""

    print("[GPU] Warming up GPU (dummy forward pass)...")
    try:
        dummy = Image.new("RGB", (64, 64), color=(128, 128, 128))
        future = inference_server.submit_analyze(
            dummy,
            "Describe this image.",
            max_tokens=8,
            priority=2,
            camera_id="warmup",
        )
        future.result(timeout=30.0)
        print("[GPU] Warm-up complete — GPU at full speed")
    except Exception as exc:
        print(f"[GPU] Warm-up skipped ({exc}) — first inference may be slower")


def build_startup_artifacts(
    config,
    camera_manager: CameraManager,
    metrics_logger,
) -> StartupArtifacts:
    """Create startup services for the distributed multi-camera runtime."""

    print("\n[AI] Initializing distributed AI pipeline...")

    edge_vision, cloud_ai, a2a_client = _init_models(config)
    fcm_notifier = _init_fcm(config)

    effective_max_rounds = config.get_effective_max_rounds()
    if effective_max_rounds != config.MAX_INVESTIGATION_ROUNDS:
        print(
            "[AI] Multi-camera throughput mode enabled "
            f"(max rounds capped at {effective_max_rounds})"
        )

    inference_server = None
    if edge_vision is not None:
        inference_server = InferenceServer(
            edge_vision,
            max_queue_size=config.GPU_QUEUE_SIZE,
        )
        inference_server.start()
        _warmup_gpu(inference_server)

    pipelines: Dict[str, CameraPipeline] = {}

    for cam_id, cam in camera_manager.cameras.items():
        def make_frame_provider(camera_id=cam_id):
            def provider():
                camera = camera_manager.get_camera(camera_id)
                if camera is None:
                    return None, None
                return camera.frame_buffer.get_current_frame()

            return provider

        pipelines[cam_id] = CameraPipeline(
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

    return StartupArtifacts(
        inference_server=inference_server,
        pipelines=pipelines,
        a2a_client=a2a_client,
    )
