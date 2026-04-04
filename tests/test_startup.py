from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from _stubs import install_dependency_stubs

install_dependency_stubs()

import app.bootstrap as bootstrap
import app.orchestrator as orchestrator
import main


class FakeConfig:
    MOTION_THRESHOLD = 2000
    CAMERA_COOLDOWN = 2.0
    MIN_CONSECUTIVE_MOTION_FRAMES = 2
    MOTION_BLUR_KERNEL = (5, 5)
    MOTION_BINARY_THRESHOLD = 20
    MOTION_DILATE_ITERATIONS = 3
    GPU_QUEUE_SIZE = 8
    MAX_INVESTIGATION_ROUNDS = 3
    EDGE_INITIAL_MAX_TOKENS = 24
    EDGE_FOLLOWUP_MAX_TOKENS = 16
    SAVE_ALERTS = True
    ALERTS_DIR = Path("alerts")
    BUFFER_DURATION_SECONDS = 10
    ENABLE_FCM_NOTIFICATIONS = False
    ENABLE_REVIEW_NOTIFICATIONS = False
    ENABLE_A2A = False
    ENABLE_RAG = True
    CLOUD_BACKEND = "groq"
    GROQ_API_KEY = "secret"
    CLOUD_MODEL_ID = "llama-3.3-70b-versatile"
    EDGE_MODEL_ID = "vikhyatk/moondream2"
    EDGE_QUANT_MODE = "auto"
    DEVICE = "cpu"
    EDGE_AGENT_URL = "http://localhost:8001"
    CLOUD_AGENT_URL = "http://localhost:8002"
    RAG_AGENT_URL = "http://localhost:8003"
    FIREBASE_CREDENTIALS_PATH = Path("firebase_key.json")
    FCM_TOPIC = "violence_alerts"

    @classmethod
    def get_cameras(cls):
        return {"cam1": "videos/a.mp4", "cam2": "videos/b.mp4"}

    @classmethod
    def get_camera_runtime_settings(cls):
        return {
            "cam1": {
                "motion_threshold": 2000,
                "cooldown_seconds": 2.0,
                "min_consecutive_motion_frames": 2,
            },
            "cam2": {
                "motion_threshold": 2100,
                "cooldown_seconds": 2.5,
                "min_consecutive_motion_frames": 3,
            },
        }

    @classmethod
    def get_effective_max_rounds(cls):
        return 2


class FakeFrameBuffer:
    def get_current_frame(self):
        return None, None


class FakeCamera:
    def __init__(self, camera_id, video_path, fps=30.0):
        self.camera_id = camera_id
        self.video_path = video_path
        self.fps = fps
        self.frame_buffer = FakeFrameBuffer()


class FakeCameraManager:
    def __init__(self):
        self.cameras = {
            "cam1": FakeCamera("cam1", "videos/a.mp4", fps=24.0),
            "cam2": FakeCamera("cam2", "videos/b.mp4", fps=30.0),
        }

    def get_camera(self, camera_id):
        return self.cameras.get(camera_id)


class StartupTests(unittest.TestCase):
    def test_main_exits_cleanly_when_config_is_invalid(self):
        with patch("main.Config.validate", return_value=False), patch(
            "main.Config.print_info"
        ) as mock_print_info, patch("main.run_application") as mock_run:
            with self.assertRaises(SystemExit) as exc:
                main.main()

        self.assertEqual(exc.exception.code, 1)
        mock_print_info.assert_not_called()
        mock_run.assert_not_called()

    def test_build_startup_artifacts_assembles_pipelines_without_real_models(self):
        camera_manager = FakeCameraManager()
        metrics_logger = MagicMock()
        edge_vision = object()
        cloud_ai = object()
        server_instance = MagicMock()
        pipeline_instances = [MagicMock(name="cam1-pipeline"), MagicMock(name="cam2-pipeline")]

        with patch(
            "app.bootstrap._init_models",
            return_value=(edge_vision, cloud_ai, None),
        ), patch(
            "app.bootstrap._init_fcm",
            return_value=None,
        ), patch(
            "app.bootstrap.InferenceServer",
            return_value=server_instance,
        ) as mock_server, patch(
            "app.bootstrap._warmup_gpu"
        ) as mock_warmup, patch(
            "app.bootstrap.CameraPipeline",
            side_effect=pipeline_instances,
        ) as mock_pipeline:
            artifacts = bootstrap.build_startup_artifacts(
                FakeConfig,
                camera_manager,
                metrics_logger,
            )

        mock_server.assert_called_once_with(edge_vision, max_queue_size=FakeConfig.GPU_QUEUE_SIZE)
        server_instance.start.assert_called_once_with()
        mock_warmup.assert_called_once_with(server_instance)
        self.assertEqual(list(artifacts.pipelines.keys()), ["cam1", "cam2"])
        self.assertIs(artifacts.pipelines["cam1"], pipeline_instances[0])
        self.assertIs(artifacts.pipelines["cam2"], pipeline_instances[1])
        self.assertEqual(mock_pipeline.call_count, 2)
        self.assertEqual(
            mock_pipeline.call_args_list[0].kwargs["video_file"],
            "cam1:a.mp4",
        )
        self.assertEqual(
            mock_pipeline.call_args_list[1].kwargs["video_file"],
            "cam2:b.mp4",
        )

    def test_run_application_exports_metrics_after_display_returns(self):
        runtime = SimpleNamespace(
            stop_event=threading.Event(),
            models_ready=threading.Event(),
            metrics_logger=MagicMock(),
        )
        camera_manager = MagicMock()
        pipelines = {
            "cam1": MagicMock(),
            "cam2": MagicMock(),
        }
        threads = [MagicMock(), MagicMock()]
        inference_server = MagicMock()
        inference_server.get_stats.return_value = {
            "requests_processed": 1,
            "encode_calls": 1,
            "answer_calls": 1,
            "errors": 0,
            "total_gpu_time_ms": 15.0,
        }
        startup = bootstrap.StartupArtifacts(
            inference_server=inference_server,
            pipelines=pipelines,
        )

        with patch("app.orchestrator.AppRuntime", return_value=runtime), patch(
            "app.orchestrator.build_camera_manager",
            return_value=camera_manager,
        ), patch(
            "app.orchestrator.build_startup_artifacts",
            return_value=startup,
        ), patch(
            "app.orchestrator.start_capture_threads",
            return_value=threads,
        ), patch("app.orchestrator.display_loop") as mock_display:
            orchestrator.run_application(FakeConfig)

        mock_display.assert_called_once_with(camera_manager, runtime)
        self.assertTrue(runtime.models_ready.is_set())
        for pipeline in pipelines.values():
            pipeline.start.assert_called_once_with(runtime.stop_event)
            pipeline.stop.assert_called_once_with(timeout=3)
        camera_manager.stop_all.assert_called_once_with()
        for thread in threads:
            thread.join.assert_called_once_with(timeout=3)
        inference_server.get_stats.assert_called_once_with()
        inference_server.shutdown.assert_called_once_with()
        runtime.metrics_logger.print_summary.assert_called_once_with()
        runtime.metrics_logger.export_json.assert_called_once_with(
            "alerts/benchmark_metrics.json"
        )
        runtime.metrics_logger.export_csv.assert_called_once_with(
            "alerts/benchmark_metrics.csv"
        )

    def test_run_application_handles_keyboard_interrupt_and_still_shuts_down(self):
        runtime = SimpleNamespace(
            stop_event=threading.Event(),
            models_ready=threading.Event(),
            metrics_logger=MagicMock(),
        )
        camera_manager = MagicMock()
        pipelines = {"cam1": MagicMock()}
        threads = [MagicMock()]
        startup = bootstrap.StartupArtifacts(
            inference_server=None,
            pipelines=pipelines,
        )

        with patch("app.orchestrator.AppRuntime", return_value=runtime), patch(
            "app.orchestrator.build_camera_manager",
            return_value=camera_manager,
        ), patch(
            "app.orchestrator.build_startup_artifacts",
            return_value=startup,
        ), patch(
            "app.orchestrator.start_capture_threads",
            return_value=threads,
        ), patch(
            "app.orchestrator.display_loop",
            side_effect=KeyboardInterrupt,
        ):
            orchestrator.run_application(FakeConfig)

        self.assertTrue(runtime.stop_event.is_set())
        camera_manager.stop_all.assert_called_once_with()
        pipelines["cam1"].stop.assert_called_once_with(timeout=3)
        threads[0].join.assert_called_once_with(timeout=3)
        runtime.metrics_logger.export_json.assert_called_once_with(
            "alerts/benchmark_metrics.json"
        )
        runtime.metrics_logger.export_csv.assert_called_once_with(
            "alerts/benchmark_metrics.csv"
        )


if __name__ == "__main__":
    unittest.main()
