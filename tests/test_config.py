import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from _stubs import install_dependency_stubs

install_dependency_stubs()


def load_config_module(env):
    sys.modules.pop("config", None)
    with patch.dict(os.environ, env, clear=True):
        with patch("dotenv.load_dotenv", return_value=False):
            return importlib.import_module("config")


class ConfigFacadeTests(unittest.TestCase):
    def test_default_single_camera_fallback_uses_cam0(self):
        module = load_config_module({"VIDEO_PATH": "videos/custom.mp4"})

        self.assertEqual(module.Config.get_cameras(), {"cam0": "videos/custom.mp4"})

    def test_camera_sources_support_rtsp_urls(self):
        module = load_config_module(
            {
                "CAMERA_SOURCES": (
                    "cam1:videos/front.mp4,"
                    "cam2:rtsp://example.com:8554/live"
                )
            }
        )

        self.assertEqual(
            module.Config.get_cameras(),
            {
                "cam1": "videos/front.mp4",
                "cam2": "rtsp://example.com:8554/live",
            },
        )

    def test_camera_overrides_merge_with_defaults_and_ignore_bad_entries(self):
        module = load_config_module(
            {
                "CAMERA_SOURCES": "cam1:videos/a.mp4,cam2:videos/b.mp4",
                "MOTION_THRESHOLD": "2000",
                "CAMERA_COOLDOWN": "2.0",
                "MIN_CONSECUTIVE_MOTION_FRAMES": "2",
                "CAMERA_OVERRIDES": (
                    "cam1:threshold=2500,cooldown=2.5,min_motion_frames=3;"
                    "bad-entry;"
                    "cam2:unknown=1,min_motion_frames=4;"
                    "cam3:threshold=1000"
                ),
            }
        )

        self.assertEqual(
            module.Config.get_camera_runtime_settings(),
            {
                "cam1": {
                    "motion_threshold": 2500,
                    "cooldown_seconds": 2.5,
                    "min_consecutive_motion_frames": 3,
                },
                "cam2": {
                    "motion_threshold": 2000,
                    "cooldown_seconds": 2.0,
                    "min_consecutive_motion_frames": 4,
                },
            },
        )

    def test_effective_max_rounds_is_capped_for_multi_camera(self):
        module = load_config_module(
            {
                "CAMERA_SOURCES": "cam1:videos/a.mp4,cam2:videos/b.mp4",
                "MAX_INVESTIGATION_ROUNDS": "5",
                "MULTI_CAMERA_MAX_INVESTIGATION_ROUNDS": "2",
            }
        )

        self.assertEqual(module.Config.get_effective_max_rounds(), 2)

    def test_validate_skips_network_streams_and_rejects_missing_local_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_video = Path(temp_dir) / "clip.mp4"
            existing_video.touch()

            valid_module = load_config_module(
                {
                    "GROQ_API_KEY": "secret",
                    "CAMERA_SOURCES": (
                        f"local:{existing_video},"
                        "net:rtsp://example.com:8554/live"
                    ),
                }
            )
            self.assertTrue(valid_module.Config.validate())

            invalid_module = load_config_module(
                {
                    "GROQ_API_KEY": "secret",
                    "CAMERA_SOURCES": (
                        "missing:/definitely/not/here.mp4,"
                        "net:rtsp://example.com:8554/live"
                    ),
                }
            )
            self.assertFalse(invalid_module.Config.validate())


if __name__ == "__main__":
    unittest.main()
