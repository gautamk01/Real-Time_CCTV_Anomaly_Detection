"""Typed settings construction from environment variables."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Tuple


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default).lower()).lower() == "true"


@dataclass(frozen=True)
class PathSettings:
    base_dir: Path
    videos_dir: Path
    alerts_dir: Path
    firebase_credentials_path: Path
    rag_db_dir: Path


@dataclass(frozen=True)
class ApiSettings:
    gemini_api_key: str
    groq_api_key: str


@dataclass(frozen=True)
class ModelSettings:
    cloud_backend: str
    edge_model_id: str
    cloud_model_id: str
    edge_quant_mode: str
    edge_initial_max_tokens: int
    edge_followup_max_tokens: int
    device: str


@dataclass(frozen=True)
class CameraSettings:
    video_path: str
    camera_sources: str
    camera_cooldown: float
    min_consecutive_motion_frames: int
    multi_camera_max_investigation_rounds: int
    camera_overrides: str


@dataclass(frozen=True)
class InvestigationSettings:
    motion_threshold: int
    max_investigation_rounds: int
    gpu_queue_size: int
    max_concurrent_investigations: int
    motion_blur_kernel: Tuple[int, int]
    motion_binary_threshold: int
    motion_dilate_iterations: int


@dataclass(frozen=True)
class AlertSettings:
    save_alerts: bool
    buffer_duration_seconds: int
    cleanup_old_alerts_days: int


@dataclass(frozen=True)
class NotificationSettings:
    enable_fcm_notifications: bool
    enable_review_notifications: bool
    fcm_topic: str


@dataclass(frozen=True)
class AgentSettings:
    enable_a2a: bool
    edge_agent_url: str
    cloud_agent_url: str
    rag_agent_url: str


@dataclass(frozen=True)
class RagSettings:
    enable_rag: bool
    embedding_model: str
    top_k: int


@dataclass(frozen=True)
class AppSettings:
    paths: PathSettings
    api: ApiSettings
    model: ModelSettings
    camera: CameraSettings
    investigation: InvestigationSettings
    alerts: AlertSettings
    notifications: NotificationSettings
    agents: AgentSettings
    rag: RagSettings


def load_settings(base_dir: Path) -> AppSettings:
    """Build typed settings from the current process environment."""

    videos_dir = base_dir / "videos"
    cloud_backend = os.getenv("CLOUD_BACKEND", "groq").lower()

    return AppSettings(
        paths=PathSettings(
            base_dir=base_dir,
            videos_dir=videos_dir,
            alerts_dir=base_dir / os.getenv("ALERTS_DIR", "alerts"),
            firebase_credentials_path=(
                base_dir / os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_key.json")
            ),
            rag_db_dir=base_dir / os.getenv("RAG_DB_DIR", "rag_db"),
        ),
        api=ApiSettings(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
        ),
        model=ModelSettings(
            cloud_backend=cloud_backend,
            edge_model_id="vikhyatk/moondream2",
            cloud_model_id=os.getenv(
                "CLOUD_MODEL_ID",
                "qwen3:8b" if cloud_backend == "ollama" else "llama-3.3-70b-versatile",
            ),
            edge_quant_mode=os.getenv("EDGE_QUANT_MODE", "auto"),
            edge_initial_max_tokens=int(os.getenv("EDGE_INITIAL_MAX_TOKENS", "24")),
            edge_followup_max_tokens=int(
                os.getenv("EDGE_FOLLOWUP_MAX_TOKENS", "16")
            ),
            device=os.getenv("DEVICE", "auto"),
        ),
        camera=CameraSettings(
            video_path=os.getenv("VIDEO_PATH", str(videos_dir / "test_video.mp4")),
            camera_sources=os.getenv("CAMERA_SOURCES", ""),
            camera_cooldown=float(os.getenv("CAMERA_COOLDOWN", "2.0")),
            min_consecutive_motion_frames=int(
                os.getenv("MIN_CONSECUTIVE_MOTION_FRAMES", "2")
            ),
            multi_camera_max_investigation_rounds=int(
                os.getenv("MULTI_CAMERA_MAX_INVESTIGATION_ROUNDS", "2")
            ),
            camera_overrides=os.getenv("CAMERA_OVERRIDES", ""),
        ),
        investigation=InvestigationSettings(
            motion_threshold=int(os.getenv("MOTION_THRESHOLD", "2000")),
            max_investigation_rounds=int(
                os.getenv("MAX_INVESTIGATION_ROUNDS", "3")
            ),
            gpu_queue_size=int(os.getenv("GPU_QUEUE_SIZE", "32")),
            max_concurrent_investigations=int(
                os.getenv("MAX_CONCURRENT_INVESTIGATIONS", "0")
            ),
            motion_blur_kernel=(5, 5),
            motion_binary_threshold=20,
            motion_dilate_iterations=3,
        ),
        alerts=AlertSettings(
            save_alerts=_env_bool("SAVE_ALERTS", True),
            buffer_duration_seconds=int(os.getenv("BUFFER_DURATION_SECONDS", "10")),
            cleanup_old_alerts_days=int(os.getenv("CLEANUP_OLD_ALERTS_DAYS", "7")),
        ),
        notifications=NotificationSettings(
            enable_fcm_notifications=_env_bool("ENABLE_FCM_NOTIFICATIONS", False),
            enable_review_notifications=_env_bool(
                "ENABLE_REVIEW_NOTIFICATIONS",
                False,
            ),
            fcm_topic=os.getenv("FCM_TOPIC", "violence_alerts"),
        ),
        agents=AgentSettings(
            enable_a2a=_env_bool("ENABLE_A2A", False),
            edge_agent_url=os.getenv("EDGE_AGENT_URL", "http://localhost:8001"),
            cloud_agent_url=os.getenv("CLOUD_AGENT_URL", "http://localhost:8002"),
            rag_agent_url=os.getenv("RAG_AGENT_URL", "http://localhost:8003"),
        ),
        rag=RagSettings(
            enable_rag=_env_bool("ENABLE_RAG", True),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            top_k=int(os.getenv("RAG_TOP_K", "3")),
        ),
    )
