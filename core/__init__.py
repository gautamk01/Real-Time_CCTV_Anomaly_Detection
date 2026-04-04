"""Core Detection Logic.

Sub-packages:
    detection/      — Frame buffering, motion detection, camera management
    investigation/  — AI investigator, GPU inference server, camera pipeline
    output/         — Alert saving, FCM notifications, metrics, evaluation
"""

from .detection.frame_buffer import FrameBuffer
from .detection.motion_detector import MotionDetector
from .detection.motion_queue import MotionQueue
from .detection.camera_manager import CameraManager, CameraSource
from .investigation.investigator import AIInvestigator
from .investigation.inference_server import InferenceServer, RequestPriority
from .investigation.camera_pipeline import CameraPipeline
from .output.metrics_logger import MetricsLogger
from .output.evaluation import ConfusionMatrix
from .output import alert_saver
from .output import fcm_notifier

__all__ = [
    "FrameBuffer",
    "MotionDetector",
    "MotionQueue",
    "CameraManager",
    "CameraSource",
    "AIInvestigator",
    "InferenceServer",
    "RequestPriority",
    "CameraPipeline",
    "MetricsLogger",
    "ConfusionMatrix",
    "alert_saver",
    "fcm_notifier",
]
