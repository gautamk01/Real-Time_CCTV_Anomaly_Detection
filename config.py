"""Compatibility facade for environment-driven configuration."""

from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from app.config import (
    CameraRuntimeDefaults,
    parse_camera_runtime_settings,
    parse_camera_sources,
    print_configuration_summary,
    validate_configuration,
    load_settings,
)

load_dotenv()


def _load_app_settings():
    return load_settings(base_dir=Path(__file__).parent)


def _apply_legacy_attributes(target, settings) -> None:
    target.BASE_DIR = settings.paths.base_dir
    target.VIDEOS_DIR = settings.paths.videos_dir

    target.GEMINI_API_KEY = settings.api.gemini_api_key
    target.GROQ_API_KEY = settings.api.groq_api_key

    target.CLOUD_BACKEND = settings.model.cloud_backend
    target.EDGE_MODEL_ID = settings.model.edge_model_id
    target.CLOUD_MODEL_ID = settings.model.cloud_model_id
    target.EDGE_QUANT_MODE = settings.model.edge_quant_mode
    target.EDGE_INITIAL_MAX_TOKENS = settings.model.edge_initial_max_tokens
    target.EDGE_FOLLOWUP_MAX_TOKENS = settings.model.edge_followup_max_tokens
    target.DEVICE = settings.model.device

    target.VIDEO_PATH = settings.camera.video_path
    target.CAMERA_SOURCES = settings.camera.camera_sources
    target.CAMERA_COOLDOWN = settings.camera.camera_cooldown
    target.MIN_CONSECUTIVE_MOTION_FRAMES = (
        settings.camera.min_consecutive_motion_frames
    )
    target.MULTI_CAMERA_MAX_INVESTIGATION_ROUNDS = (
        settings.camera.multi_camera_max_investigation_rounds
    )
    target.CAMERA_OVERRIDES = settings.camera.camera_overrides

    target.MOTION_THRESHOLD = settings.investigation.motion_threshold
    target.MAX_INVESTIGATION_ROUNDS = settings.investigation.max_investigation_rounds
    target.GPU_QUEUE_SIZE = settings.investigation.gpu_queue_size
    target.MAX_CONCURRENT_INVESTIGATIONS = (
        settings.investigation.max_concurrent_investigations
    )
    target.MOTION_BLUR_KERNEL = settings.investigation.motion_blur_kernel
    target.MOTION_BINARY_THRESHOLD = settings.investigation.motion_binary_threshold
    target.MOTION_DILATE_ITERATIONS = settings.investigation.motion_dilate_iterations

    target.SAVE_ALERTS = settings.alerts.save_alerts
    target.ALERTS_DIR = settings.paths.alerts_dir
    target.BUFFER_DURATION_SECONDS = settings.alerts.buffer_duration_seconds
    target.CLEANUP_OLD_ALERTS_DAYS = settings.alerts.cleanup_old_alerts_days

    target.ENABLE_FCM_NOTIFICATIONS = (
        settings.notifications.enable_fcm_notifications
    )
    target.ENABLE_REVIEW_NOTIFICATIONS = (
        settings.notifications.enable_review_notifications
    )
    target.FIREBASE_CREDENTIALS_PATH = settings.paths.firebase_credentials_path
    target.FCM_TOPIC = settings.notifications.fcm_topic

    target.ENABLE_A2A = settings.agents.enable_a2a
    target.EDGE_AGENT_URL = settings.agents.edge_agent_url
    target.CLOUD_AGENT_URL = settings.agents.cloud_agent_url
    target.RAG_AGENT_URL = settings.agents.rag_agent_url

    target.ENABLE_RAG = settings.rag.enable_rag
    target.RAG_DB_DIR = settings.paths.rag_db_dir
    target.RAG_EMBEDDING_MODEL = settings.rag.embedding_model
    target.RAG_TOP_K = settings.rag.top_k


class Config:
    """Legacy configuration facade backed by typed helper modules."""

    @classmethod
    def get_cameras(cls) -> Dict[str, str]:
        """Return the configured camera source mapping."""

        return parse_camera_sources(cls.CAMERA_SOURCES, cls.VIDEO_PATH)

    @classmethod
    def get_camera_runtime_settings(cls) -> Dict[str, Dict[str, float]]:
        """Return per-camera runtime settings with defaults and overrides."""

        defaults = CameraRuntimeDefaults(
            motion_threshold=cls.MOTION_THRESHOLD,
            cooldown_seconds=cls.CAMERA_COOLDOWN,
            min_consecutive_motion_frames=cls.MIN_CONSECUTIVE_MOTION_FRAMES,
        )
        return parse_camera_runtime_settings(
            cls.CAMERA_OVERRIDES,
            cls.get_cameras().keys(),
            defaults,
        )

    @classmethod
    def get_effective_max_rounds(cls) -> int:
        """Clamp investigation rounds in multi-camera mode."""

        if len(cls.get_cameras()) > 1:
            return min(
                cls.MAX_INVESTIGATION_ROUNDS,
                cls.MULTI_CAMERA_MAX_INVESTIGATION_ROUNDS,
            )
        return cls.MAX_INVESTIGATION_ROUNDS

    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration values."""

        errors = validate_configuration(
            groq_api_key=cls.GROQ_API_KEY,
            cameras=cls.get_cameras(),
        )
        if not errors:
            return True

        print("\n" + "=" * 60)
        print("🔴 CONFIGURATION ERRORS")
        print("=" * 60)
        for error in errors:
            print(error)
        print("\nPlease check your .env file and video path.")
        print("=" * 60 + "\n")
        return False

    @classmethod
    def print_info(cls) -> None:
        """Print configuration information."""

        cameras = cls.get_cameras()
        camera_settings = cls.get_camera_runtime_settings()
        print_configuration_summary(
            cls,
            cameras,
            camera_settings,
            cls.get_effective_max_rounds(),
        )


_SETTINGS = _load_app_settings()
Config._settings = _SETTINGS
_apply_legacy_attributes(Config, _SETTINGS)
