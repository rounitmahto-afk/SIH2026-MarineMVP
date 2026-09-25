from .detector import MarineDetector, load_default_detector
from .schema import Detection, DetectionResult

__all__ = [
    "MarineDetector",
    "Detection",
    "DetectionResult",
    "load_default_detector",
]
