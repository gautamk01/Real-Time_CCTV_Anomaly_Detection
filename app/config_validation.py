"""Validation and presentation helpers for configuration."""

from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse


def validate_configuration(
    groq_api_key: str,
    cameras: Dict[str, str],
) -> List[str]:
    """Return a list of configuration validation errors."""

    errors = []

    if not groq_api_key:
        errors.append("❌ GROQ_API_KEY not set in .env file")

    for cam_id, path in cameras.items():
        parsed = urlparse(path)
        if parsed.scheme in ("rtsp", "rtsps", "http", "https"):
            continue
        if not Path(path).exists():
            errors.append(f"❌ Video not found for {cam_id}: {path}")

    return errors


def resolve_device(device: str) -> str:
    """Resolve the effective runtime device string."""

    if device != "auto":
        return device

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def print_configuration_summary(
    config,
    cameras: Dict[str, str],
    camera_settings: Dict[str, Dict[str, float]],
    effective_max_rounds: int,
) -> None:
    """Print a readable configuration summary."""

    device = resolve_device(config.DEVICE)

    print("\n" + "=" * 60)
    print("⚙️  CONFIGURATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Edge Model: {config.EDGE_MODEL_ID}")
    print(f"Edge Quantization: {config.EDGE_QUANT_MODE}")
    print(
        "Edge Token Budgets: "
        f"initial={config.EDGE_INITIAL_MAX_TOKENS}, "
        f"follow-up={config.EDGE_FOLLOWUP_MAX_TOKENS}"
    )
    print(f"Cloud Model: {config.CLOUD_MODEL_ID}")

    if len(cameras) == 1 and "cam0" in cameras:
        print(f"Video: {Path(config.VIDEO_PATH).name}")
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
        print(f"Base Motion Threshold: {config.MOTION_THRESHOLD}")
        print(
            "Base Motion Gating: "
            f"{config.MIN_CONSECUTIVE_MOTION_FRAMES} consecutive frames"
        )
        print(f"Base Camera Cooldown: {config.CAMERA_COOLDOWN:.1f}s")
        override_count = sum(
            1
            for cfg in camera_settings.values()
            if int(cfg["motion_threshold"]) != config.MOTION_THRESHOLD
            or float(cfg["cooldown_seconds"]) != config.CAMERA_COOLDOWN
            or int(cfg["min_consecutive_motion_frames"])
            != config.MIN_CONSECUTIVE_MOTION_FRAMES
        )
        if override_count:
            print(f"Camera Overrides: {override_count} camera(s)")

    if effective_max_rounds != config.MAX_INVESTIGATION_ROUNDS:
        print(
            "Max Investigation Rounds: "
            f"{effective_max_rounds} "
            f"(multi-camera cap from {config.MAX_INVESTIGATION_ROUNDS})"
        )
    else:
        print(f"Max Investigation Rounds: {config.MAX_INVESTIGATION_ROUNDS}")

    print(f"Save Alerts: {'Yes' if config.SAVE_ALERTS else 'No'}")
    if config.SAVE_ALERTS:
        print(f"Alerts Directory: {config.ALERTS_DIR}")
        print(f"Buffer Duration: {config.BUFFER_DURATION_SECONDS}s")

    print(f"A2A Mode: {'Networked' if config.ENABLE_A2A else 'Direct (local)'}")
    if config.ENABLE_A2A:
        print(f"Edge Agent: {config.EDGE_AGENT_URL}")
        print(f"Cloud Agent: {config.CLOUD_AGENT_URL}")
        print(f"RAG Agent: {config.RAG_AGENT_URL}")

    print(f"RAG: {'Enabled' if config.ENABLE_RAG else 'Disabled'}")
    print("=" * 60 + "\n")
