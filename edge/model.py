"""
PyTorch Edge Feature Extractor for Surveillance Video Clips.

Mandatory PyTorch Implementation (No Deterministic Fallback):
  - Requires PyTorch and torchvision with pretrained backbones:
      * MobileNetV3-Small (default, 576-dim feature vector)
      * EfficientNet-B0 (1280-dim feature vector)
  - Applies standard ImageNet normalization transforms.
  - Computes per-frame deep feature representations via @torch.no_grad().
  - Performs temporal average-pooling across frames -> single 1-D vector.
  - Projects to target embedding dimension if configured.
  - L2-normalizes embedding for cosine similarity in Qdrant Edge.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Union

import numpy as np
import torch
import torchvision.models as models  # type: ignore[import-not-found,import-untyped]
import torchvision.transforms as T  # type: ignore[import-not-found,import-untyped]

from edge.config import EdgeConfig, config as default_config
from edge.segmenter import VideoClip

logger = logging.getLogger(__name__)

# Standard ImageNet preprocessing
_transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

SUPPORTED_MODELS = ("mobilenet_v3", "mobilenet_v3_small", "mobilenet", "efficientnet_b0", "efficientnet")


def _build_backbone(model_type: str, embedding_dim: int) -> torch.nn.Module:
    """Build the PyTorch feature-extraction backbone with target embedding dimensionality."""
    model_name = model_type.lower()
    if "mobilenet" in model_name:
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = base.classifier[0].in_features  # 1024 (pooling after features is 576)
        # Note: in MobileNetV3-Small, base.classifier[0] takes 576 features.
        # Replacing base.classifier with Identity() outputs 576.
        base_feature_dim = base.classifier[0].in_features
        if embedding_dim and embedding_dim != base_feature_dim:
            base.classifier = torch.nn.Sequential(
                torch.nn.Linear(base_feature_dim, embedding_dim),
            )
        else:
            base.classifier = torch.nn.Identity()
    elif "efficientnet" in model_name:
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = base.classifier[1].in_features  # 1280
        if embedding_dim and embedding_dim != in_features:
            base.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features, embedding_dim),
            )
        else:
            base.classifier = torch.nn.Identity()
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            f"Supported PyTorch models: {SUPPORTED_MODELS}. "
            "Deterministic or synthetic fallback has been permanently removed; "
            "a valid PyTorch backbone is strictly required."
        )

    return base


class EdgeFeatureExtractor:
    """
    PyTorch Edge Feature Extractor.
    Requires PyTorch and torchvision with pretrained weights.
    """

    def __init__(
        self,
        model_type: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        device: Optional[str] = None,
        cfg: Optional[EdgeConfig] = None,
    ) -> None:
        self.config = cfg or default_config
        self.model_type = (model_type or self.config.model_type).lower()
        self.embedding_dim = embedding_dim or self.config.embedding_dim

        if device:
            self.device_str = device
        elif torch.cuda.is_available():
            self.device_str = "cuda"
        else:
            self.device_str = "cpu"

        self.device = torch.device(self.device_str)
        logger.info("Initializing PyTorch EdgeFeatureExtractor (%s) on %s...", self.model_type, self.device)

        self.model = _build_backbone(self.model_type, self.embedding_dim).to(self.device)
        self.model.eval()
        self.last_inference_ms: float = 0.0
        logger.info("EdgeFeatureExtractor ready (embedding_dim=%d)", self.embedding_dim)

    @property
    def is_model_loaded(self) -> bool:
        return self.model is not None

    # ----------------------------------------------------------------------
    # Inference & Extraction
    # ----------------------------------------------------------------------

    def extract_from_clip(self, clip: VideoClip) -> np.ndarray:
        """Extract normalized embedding from a VideoClip object."""
        return self.extract(clip.frames_rgb)

    @torch.no_grad()
    def extract(self, frames: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Extract a single L2-normalized embedding from video frames.

        Args:
            frames: np.ndarray (T, H, W, C) or list of (H, W, C) uint8 RGB frames.
        Returns:
            1-D numpy array of shape (embedding_dim,), dtype float32, unit L2 norm.
        """
        t0 = time.perf_counter()

        if isinstance(frames, list):
            frame_list = frames
        elif isinstance(frames, np.ndarray):
            if frames.ndim == 4 and frames.shape[-1] == 3:
                frame_list = [frames[i] for i in range(frames.shape[0])]
            elif frames.ndim == 4 and frames.shape[1] == 3:
                transposed = np.transpose(frames, (0, 2, 3, 1))
                frame_list = [transposed[i] for i in range(transposed.shape[0])]
            elif frames.ndim == 3:
                frame_list = [frames]
            else:
                raise ValueError(f"Unsupported frames array shape: {frames.shape}")
        else:
            raise ValueError(f"Unsupported frames type: {type(frames)}")

        if not frame_list:
            raise ValueError("No frames provided to feature extractor")

        tensors = []
        for f in frame_list:
            arr = f
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            tensors.append(_transform(arr))

        batch = torch.stack(tensors).to(self.device)  # (T, 3, 224, 224)
        features = self.model(batch)  # (T, D)

        # Temporal average pooling across frames
        embedding = features.mean(dim=0).cpu().numpy().astype(np.float32).reshape(-1)

        # Validate embedding dimensionality matches configured dimension
        if embedding.shape[0] != self.embedding_dim:
            raise RuntimeError(
                f"Feature extractor output dimension mismatch: got {embedding.shape[0]}, "
                f"expected {self.embedding_dim}. No dimension slicing/padding fallback is permitted."
            )

        # L2-normalize for cosine similarity in Qdrant Edge
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding /= norm
        else:
            embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            embedding[0] = 1.0

        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        return embedding

    def extract_batch(self, clips: List[VideoClip]) -> np.ndarray:
        """Extract embeddings for multiple clips, returning (N, embedding_dim)."""
        if not clips:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        embeddings = [self.extract_from_clip(c) for c in clips]
        return np.stack(embeddings, axis=0)
