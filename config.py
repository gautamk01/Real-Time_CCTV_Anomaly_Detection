"""Configuration Management for Violence Detection System."""

import os
import logging
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration manager using environment variables."""

    # Project paths
    BASE_DIR = Path(__file__).parent
    VIDEOS_DIR = BASE_DIR / "videos"

    # API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    # Cloud Backend: "groq" or "ollama"
    CLOUD_BACKEND = os.getenv("CLOUD_BACKEND", "groq").lower()

    # Model Configuration
    EDGE_MODEL_ID = "vikhyatk/moondream2"
    CLOUD_MODEL_ID = os.getenv(
        "CLOUD_MODEL_ID",
        "qwen3:8b" if os.getenv("CLOUD_BACKEND", "groq").lower() == "ollama"
        else "llama-3.3-70b-versatile",
    )
    EDGE_QUANT_MODE = os.getenv("EDGE_QUANT_MODE", "auto")
    EDGE_INITIAL_MAX_TOKENS = int(os.getenv("EDGE_INITIAL_MAX_TOKENS", "24"))
    EDGE_FOLLOWUP_MAX_TOKENS = int(os.getenv("EDGE_FOLLOWUP_MAX_TOKENS", "16"))
    
    # Device Configuration
    DEVICE = os.getenv("DEVICE", "auto")  # "cuda", "cpu", or "auto"
    
    # Video Configuration
    VIDEO_PATH = os.getenv("VIDEO_PATH", str(VIDEOS_DIR / "test_video.mp4"))

    # Multi-Camera Configuration
    # Format: comma-separated "id:path" pairs
    # e.g., "cam1:/dev/video0,cam2:rtsp://192.168.1.100/stream"
    # If empty, falls back to single VIDEO_PATH as "cam0"
    CAMERA_SOURCES = os.getenv("CAMERA_SOURCES", "")

    # Per-camera cooldown after investigation (seconds)
    CAMERA_COOLDOWN = float(os.getenv("CAMERA_COOLDOWN", "2.0"))
    MIN_CONSECUTIVE_MOTION_FRAMES = int(
        os.getenv("MIN_CONSECUTIVE_MOTION_FRAMES", "2")
    )
    MULTI_CAMERA_MAX_INVESTIGATION_ROUNDS = int(
        os.getenv("MULTI_CAMERA_MAX_INVESTIGATION_ROUNDS", "2")
    )
    # Format:
    #   cam1:threshold=2500,cooldown=2.5,min_motion_frames=3;cam2:threshold=1800
    CAMERA_OVERRIDES = os.getenv("CAMERA_OVERRIDES", "")

    # Detection Parameters
    MOTION_THRESHOLD = int(os.getenv("MOTION_THRESHOLD", "2000"))
    MAX_INVESTIGATION_ROUNDS = int(os.getenv("MAX_INVESTIGATION_ROUNDS", "3"))

    # GPU Inference Server (Distributed Architecture)
    GPU_QUEUE_SIZE = int(os.getenv("GPU_QUEUE_SIZE", "32"))
    MAX_CONCURRENT_INVESTIGATIONS = int(
        os.getenv("MAX_CONCURRENT_INVESTIGATIONS", "0")
    )  # 0 = one per camera (default)

    # Alert Saving
    SAVE_ALERTS = os.getenv("SAVE_ALERTS", "true").lower() == "true"
    ALERTS_DIR = BASE_DIR / os.getenv("ALERTS_DIR", "alerts")
    BUFFER_DURATION_SECONDS = int(os.getenv("BUFFER_DURATION_SECONDS", "10"))
    CLEANUP_OLD_ALERTS_DAYS = int(os.getenv("CLEANUP_OLD_ALERTS_DAYS", "7"))
    
    # Firebase Cloud Messaging
    ENABLE_FCM_NOTIFICATIONS = os.getenv("ENABLE_FCM_NOTIFICATIONS", "false").lower() == "true"
    ENABLE_REVIEW_NOTIFICATIONS = os.getenv("ENABLE_REVIEW_NOTIFICATIONS", "false").lower() == "true"
    FIREBASE_CREDENTIALS_PATH = BASE_DIR / os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_key.json")
    FCM_TOPIC = os.getenv("FCM_TOPIC", "violence_alerts")

    
    # A2A (Agent-to-Agent) Communication
    ENABLE_A2A = os.getenv("ENABLE_A2A", "false").lower() == "true"
    EDGE_AGENT_URL = os.getenv("EDGE_AGENT_URL", "http://localhost:8001")
    CLOUD_AGENT_URL = os.getenv("CLOUD_AGENT_URL", "http://localhost:8002")
    RAG_AGENT_URL = os.getenv("RAG_AGENT_URL", "http://localhost:8003")

    # RAG (Retrieval-Augmented Generation)
    ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() == "true"
    RAG_DB_DIR = BASE_DIR / os.getenv("RAG_DB_DIR", "rag_db")
    RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

    # Frame Processing
    MOTION_BLUR_KERNEL = (5, 5)
    MOTION_BINARY_THRESHOLD = 20
    MOTION_DILATE_ITERATIONS = 3
    
    @classmethod
    def get_cameras(cls) -> Dict[str, str]:
        """Return {camera_id: video_path} mapping.

        If CAMERA_SOURCES is set, parses it. Otherwise falls back
        to single VIDEO_PATH as 'cam0' for backward compatibility.

        Format: comma-separated "id:path" pairs.
        Uses first colon as delimiter so RTSP paths like
        cam1:rtsp://host:8554/stream work correctly.
        """
        if cls.CAMERA_SOURCES:
            cameras: Dict[str, str] = {}
            for entry in cls.CAMERA_SOURCES.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                if ":" not in entry:
                    logging.warning(
                        "Config: ignoring malformed CAMERA_SOURCES entry %r "
                        "(expected 'id:path')", entry,
                    )
                    continue
                cam_id, path = entry.split(":", 1)
                cameras[cam_id.strip()] = path.strip()
            return cameras
        return {"cam0": cls.VIDEO_PATH}

    @classmethod
    def get_camera_runtime_settings(cls) -> Dict[str, Dict[str, float]]:
        """Return per-camera runtime settings with defaults + overrides."""
        settings: Dict[str, Dict[str, float]] = {
            cam_id: {
                "motion_threshold": cls.MOTION_THRESHOLD,
                "cooldown_seconds": cls.CAMERA_COOLDOWN,
                "min_consecutive_motion_frames": cls.MIN_CONSECUTIVE_MOTION_FRAMES,
            }
            for cam_id in cls.get_cameras()
        }

        raw_overrides = cls.CAMERA_OVERRIDES.strip()
        if not raw_overrides:
            return settings

        for camera_chunk in raw_overrides.split(";"):
            camera_chunk = camera_chunk.strip()
            if not camera_chunk:
                continue
            if ":" not in camera_chunk:
                logging.warning(
                    "Config: ignoring malformed CAMERA_OVERRIDES entry %r",
                    camera_chunk,
                )
                continue

            cam_id, overrides_text = camera_chunk.split(":", 1)
            cam_id = cam_id.strip()
            if cam_id not in settings:
                logging.warning(
                    "Config: ignoring CAMERA_OVERRIDES for unknown camera %r",
                    cam_id,
                )
                continue

            for item in overrides_text.split(","):
                item = item.strip()
                if not item:
                    continue
                if "=" not in item:
                    logging.warning(
                        "Config: ignoring malformed override %r for %s",
                        item,
                        cam_id,
                    )
                    continue

                key, value = item.split("=", 1)
                key = key.strip().lower()
                value = value.strip()

                try:
                    if key in ("threshold", "motion_threshold"):
                        settings[cam_id]["motion_threshold"] = int(value)
                    elif key in ("cooldown", "cooldown_seconds"):
                        settings[cam_id]["cooldown_seconds"] = float(value)
                    elif key in (
                        "min_motion_frames",
                        "min_consecutive_motion_frames",
                    ):
                        settings[cam_id]["min_consecutive_motion_frames"] = int(value)
                    else:
                        logging.warning(
                            "Config: ignoring unknown CAMERA_OVERRIDES key %r for %s",
                            key,
                            cam_id,
                        )
                except ValueError:
                    logging.warning(
                        "Config: ignoring invalid CAMERA_OVERRIDES value %r for %s:%s",
                        value,
                        cam_id,
                        key,
                    )

        return settings

    @classmethod
    def get_effective_max_rounds(cls) -> int:
        """Clamp rounds in multi-camera mode to protect throughput."""
        if len(cls.get_cameras()) > 1:
            return min(
                cls.MAX_INVESTIGATION_ROUNDS,
                cls.MULTI_CAMERA_MAX_INVESTIGATION_ROUNDS,
            )
        return cls.MAX_INVESTIGATION_ROUNDS

    @classmethod
    def validate(cls):
        """Validate critical configuration values."""
        errors = []
        
        if not cls.GROQ_API_KEY:
            errors.append("❌ GROQ_API_KEY not set in .env file")
        
        cameras = cls.get_cameras()
        for cam_id, path in cameras.items():
            parsed = urlparse(path)
            if parsed.scheme in ("rtsp", "rtsps", "http", "https"):
                continue  # network stream — skip file-existence check
            if not Path(path).exists():
                errors.append(f"❌ Video not found for {cam_id}: {path}")
        
        if errors:
            print("\n" + "="*60)
            print("🔴 CONFIGURATION ERRORS")
            print("="*60)
            for error in errors:
                print(error)
            print("\nPlease check your .env file and video path.")
            print("="*60 + "\n")
            return False
        
        return True
    
    @classmethod
    def print_info(cls):
        """Print configuration information."""
        import torch
        
        device = cls.DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("\n" + "="*60)
        print("⚙️  CONFIGURATION")
        print("="*60)
        print(f"Device: {device}")
        print(f"Edge Model: {cls.EDGE_MODEL_ID}")
        print(f"Edge Quantization: {cls.EDGE_QUANT_MODE}")
        print(
            "Edge Token Budgets: "
            f"initial={cls.EDGE_INITIAL_MAX_TOKENS}, "
            f"follow-up={cls.EDGE_FOLLOWUP_MAX_TOKENS}"
        )
        print(f"Cloud Model: {cls.CLOUD_MODEL_ID}")
        cameras = cls.get_cameras()
        camera_settings = cls.get_camera_runtime_settings()
        if len(cameras) == 1 and "cam0" in cameras:
            print(f"Video: {Path(cls.VIDEO_PATH).name}")
        else:
            print(f"Cameras: {len(cameras)}")
            for cam_id, path in cameras.items():
                print(f"  {cam_id}: {Path(path).name}")
        if len(cameras) == 1:
            cam_id = next(iter(cameras))
            camera_cfg = camera_settings[cam_id]
            print(f"Motion Threshold: {int(camera_cfg['motion_threshold'])}")
            print(
                "Motion Gating: "
                f"{int(camera_cfg['min_consecutive_motion_frames'])} consecutive frames"
            )
            print(f"Camera Cooldown: {camera_cfg['cooldown_seconds']:.1f}s")
        else:
            print(f"Base Motion Threshold: {cls.MOTION_THRESHOLD}")
            print(
                "Base Motion Gating: "
                f"{cls.MIN_CONSECUTIVE_MOTION_FRAMES} consecutive frames"
            )
            print(f"Base Camera Cooldown: {cls.CAMERA_COOLDOWN:.1f}s")
            override_count = sum(
                1 for cfg in camera_settings.values()
                if int(cfg["motion_threshold"]) != cls.MOTION_THRESHOLD
                or float(cfg["cooldown_seconds"]) != cls.CAMERA_COOLDOWN
                or int(cfg["min_consecutive_motion_frames"])
                != cls.MIN_CONSECUTIVE_MOTION_FRAMES
            )
            if override_count:
                print(f"Camera Overrides: {override_count} camera(s)")
        effective_rounds = cls.get_effective_max_rounds()
        if effective_rounds != cls.MAX_INVESTIGATION_ROUNDS:
            print(
                "Max Investigation Rounds: "
                f"{effective_rounds} (multi-camera cap from {cls.MAX_INVESTIGATION_ROUNDS})"
            )
        else:
            print(f"Max Investigation Rounds: {cls.MAX_INVESTIGATION_ROUNDS}")
        print(f"Save Alerts: {'Yes' if cls.SAVE_ALERTS else 'No'}")
        if cls.SAVE_ALERTS:
            print(f"Alerts Directory: {cls.ALERTS_DIR}")
            print(f"Buffer Duration: {cls.BUFFER_DURATION_SECONDS}s")
        print(f"A2A Mode: {'Networked' if cls.ENABLE_A2A else 'Direct (local)'}")
        if cls.ENABLE_A2A:
            print(f"Edge Agent: {cls.EDGE_AGENT_URL}")
            print(f"Cloud Agent: {cls.CLOUD_AGENT_URL}")
            print(f"RAG Agent: {cls.RAG_AGENT_URL}")
        print(f"RAG: {'Enabled' if cls.ENABLE_RAG else 'Disabled'}")
        print("="*60 + "\n")
