"""Tests for EdgeFeatureExtractor using mandatory PyTorch backbones."""

import numpy as np
import pytest

from edge.model import EdgeFeatureExtractor, SUPPORTED_MODELS
from edge.segmenter import VideoClip


def test_feature_extractor_mobilenet_v3_default():
    """Test default MobileNetV3-Small backbone produces normalized 576-dim embeddings."""
    extractor = EdgeFeatureExtractor(model_type="mobilenet_v3", embedding_dim=576)

    assert extractor.embedding_dim == 576
    assert "mobilenet" in extractor.model_type

    # Create dummy frames: (T=8, H=224, W=224, C=3)
    frames = np.zeros((8, 224, 224, 3), dtype=np.uint8)
    frames[:, :, :, 0] = 120  # Constant color
    emb = extractor.extract(frames)

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (576,)
    assert emb.dtype == np.float32

    # Verify L2 normalization
    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-4)
    assert extractor.last_inference_ms >= 0.0


def test_feature_extractor_efficientnet_output():
    """Test EfficientNet-B0 backbone produces normalized 1280-dim embeddings."""
    extractor = EdgeFeatureExtractor(model_type="efficientnet_b0", embedding_dim=1280)

    assert extractor.embedding_dim == 1280
    assert "efficientnet" in extractor.model_type

    frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
    frames[:, :, :, 1] = 150
    emb = extractor.extract(frames)

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1280,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-4)


def test_feature_extractor_custom_dimension_projection():
    """Test that specifying custom embedding_dim applies linear projection layer."""
    custom_dim = 256
    extractor = EdgeFeatureExtractor(model_type="mobilenet_v3", embedding_dim=custom_dim)

    assert extractor.embedding_dim == custom_dim
    frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
    emb = extractor.extract(frames)

    assert emb.shape == (custom_dim,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-4)


def test_feature_extractor_rejects_synthetic_and_unsupported_models():
    """Test that synthetic/deterministic fallback is rejected and PyTorch models are required."""
    with pytest.raises(ValueError, match="Unsupported model_type: 'synthetic'"):
        EdgeFeatureExtractor(model_type="synthetic")

    with pytest.raises(ValueError, match="Unsupported model_type: 'non_existent_model'"):
        EdgeFeatureExtractor(model_type="non_existent_model")


def test_feature_extractor_deterministic_consistency():
    """Test that identical frame sequences produce identical embeddings in eval mode."""
    extractor = EdgeFeatureExtractor(model_type="mobilenet_v3", embedding_dim=576)

    frames1 = np.ones((8, 224, 224, 3), dtype=np.uint8) * 50
    frames2 = np.ones((8, 224, 224, 3), dtype=np.uint8) * 50

    emb1 = extractor.extract(frames1)
    emb2 = extractor.extract(frames2)

    # Identical frames produce identical embeddings under PyTorch eval
    np.testing.assert_allclose(emb1, emb2, rtol=1e-5, atol=1e-5)


def test_feature_extractor_sensitivity_to_changes():
    """Test that distinct visual patterns produce distinct embeddings."""
    extractor = EdgeFeatureExtractor(model_type="mobilenet_v3", embedding_dim=576)

    # Frame sequence 1: uniform dark gray
    frames1 = np.ones((8, 224, 224, 3), dtype=np.uint8) * 30

    # Frame sequence 2: high-contrast alternating pattern
    frames2 = np.zeros((8, 224, 224, 3), dtype=np.uint8)
    for t in range(8):
        frames2[t, 50:180, 50:180, :] = 240

    emb1 = extractor.extract(frames1)
    emb2 = extractor.extract(frames2)

    # Cosine distance should be non-zero
    cosine_sim = np.dot(emb1, emb2)
    cosine_dist = 1.0 - cosine_sim
    assert cosine_dist > 0.01


def test_feature_extractor_extract_from_clip():
    """Test extracting embedding directly from VideoClip object."""
    extractor = EdgeFeatureExtractor(model_type="mobilenet_v3", embedding_dim=576)

    dummy_frames = np.random.randint(0, 256, (8, 224, 224, 3), dtype=np.uint8)
    clip = VideoClip(
        channel=1,
        camera_name="Driveway",
        frames_rgb=dummy_frames,
        start_time=100.0,
        end_time=110.0,
        duration_s=10.0,
        raw_frame_count=100,
        sampled_frame_count=8,
    )

    emb = extractor.extract_from_clip(clip)
    assert emb.shape == (576,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-4)


def test_feature_extractor_batch():
    """Test batch extraction for multiple VideoClips."""
    extractor = EdgeFeatureExtractor(model_type="mobilenet_v3", embedding_dim=576)

    clips = []
    for ch in range(1, 4):
        dummy_frames = np.zeros((8, 224, 224, 3), dtype=np.uint8)
        dummy_frames[:, :, :, ch - 1] = 200
        clips.append(
            VideoClip(
                channel=ch,
                camera_name=f"Camera {ch}",
                frames_rgb=dummy_frames,
                start_time=100.0,
                end_time=110.0,
                duration_s=10.0,
                raw_frame_count=50,
                sampled_frame_count=8,
            )
        )

    mat = extractor.extract_batch(clips)
    assert mat.shape == (3, 576)
    for row in mat:
        assert np.isclose(np.linalg.norm(row), 1.0, atol=1e-4)
