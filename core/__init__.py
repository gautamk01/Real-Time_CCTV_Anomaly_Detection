"""Core Detection Logic."""

from .frame_buffer import FrameBuffer
from .motion_detector import MotionDetector
from .investigator import AIInvestigator
from .metrics_logger import MetricsLogger
from .evaluation import ConfusionMatrix
from .motion_queue import MotionQueue
from .camera_manager import CameraManager, CameraSource
from .inference_server import InferenceServer, RequestPriority
from .camera_pipeline import CameraPipeline
from . import alert_saver
from . import fcm_notifier

#__all__ is a gatekeeper ONLY for wildcard imports.
__all__ = [
    'FrameBuffer', 'MotionDetector', 'AIInvestigator', 'MetricsLogger',
    'ConfusionMatrix', 'MotionQueue', 'CameraManager', 'CameraSource',
    'InferenceServer', 'RequestPriority', 'CameraPipeline',
    'alert_saver', 'fcm_notifier',
]
