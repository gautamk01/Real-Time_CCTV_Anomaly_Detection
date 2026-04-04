"""Parsing helpers for environment-driven configuration."""

from dataclasses import dataclass
import logging
from typing import Dict, Iterable


@dataclass(frozen=True)
class CameraRuntimeDefaults:
    """Default per-camera runtime values."""

    motion_threshold: int
    cooldown_seconds: float
    min_consecutive_motion_frames: int


def parse_camera_sources(
    raw_sources: str,
    default_video_path: str,
) -> Dict[str, str]:
    """Return {camera_id: path} from env text or a single-camera fallback."""

    if raw_sources:
        cameras: Dict[str, str] = {}
        for entry in raw_sources.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" not in entry:
                logging.warning(
                    "Config: ignoring malformed CAMERA_SOURCES entry %r "
                    "(expected 'id:path')",
                    entry,
                )
                continue
            cam_id, path = entry.split(":", 1)
            cameras[cam_id.strip()] = path.strip()
        return cameras

    return {"cam0": default_video_path}


def parse_camera_runtime_settings(
    raw_overrides: str,
    camera_ids: Iterable[str],
    defaults: CameraRuntimeDefaults,
) -> Dict[str, Dict[str, float]]:
    """Return per-camera runtime settings with defaults and env overrides."""

    settings: Dict[str, Dict[str, float]] = {
        cam_id: {
            "motion_threshold": defaults.motion_threshold,
            "cooldown_seconds": defaults.cooldown_seconds,
            "min_consecutive_motion_frames": defaults.min_consecutive_motion_frames,
        }
        for cam_id in camera_ids
    }

    raw_overrides = raw_overrides.strip()
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
