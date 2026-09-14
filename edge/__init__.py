"""Edge surveillance package."""

from edge.config import EdgeConfig, config
from edge.capture_manager import CaptureManager
from edge.stream_worker import StreamWorker, FrameItem, StreamStats
from edge.segmenter import SlidingWindowSegmenter, VideoClip
from edge.model import EdgeFeatureExtractor
from edge.detector import QdrantEdgeDetector
from edge.queue import SQLiteOfflineQueue, IncidentItem, AutoDrainWorker
from edge.main import EdgeTriageService

__version__ = "0.2.0"

__all__ = [
    "EdgeConfig",
    "config",
    "CaptureManager",
    "StreamWorker",
    "FrameItem",
    "StreamStats",
    "SlidingWindowSegmenter",
    "VideoClip",
    "EdgeFeatureExtractor",
    "QdrantEdgeDetector",
    "SQLiteOfflineQueue",
    "IncidentItem",
    "AutoDrainWorker",
    "EdgeTriageService",
]
