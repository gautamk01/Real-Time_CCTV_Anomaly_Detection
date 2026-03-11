"""Core Detection Logic."""

from .frame_buffer import FrameBuffer
from .motion_detector import MotionDetector
from .investigator import AIInvestigator
from .metrics_logger import MetricsLogger
from .evaluation import ConfusionMatrix
from . import alert_saver
from . import fcm_notifier

#__all__ is a gatekeeper ONLY for wildcard imports.
__all__ = ['FrameBuffer', 'MotionDetector', 'AIInvestigator', 'MetricsLogger', 'ConfusionMatrix', 'alert_saver', 'fcm_notifier']
