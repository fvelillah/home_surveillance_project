"""Unit and integration tests for scripts/embed_and_index_baseline.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from scripts.embed_and_index_baseline import (
    DahuaVideoFile,
    cluster_channel_embeddings,
    discover_dahua_videos,
    index_channel_centroids,
    parse_dahua_filename,
    run_baseline_pipeline,
)


def test_parse_dahua_filename_standard():
    """Tests parsing canonical Dahua NVR naming schema."""
    filename = "NVR_ch01_extra1_20260919120000_20260919130000.asf"
    res = parse_dahua_filename(filename)
    assert res is not None
    assert res.channel == 1
    assert res.stream_tier == "extra1"
    assert res.is_substream is True
    assert res.start_time_str == "20260919120000"
    assert res.end_time_str == "20260919130000"


def test_parse_dahua_filename_mainstream_and_variations():
    """Tests parsing mainstream and shortened prefix variations."""
    f1 = "ch04_main_20260919140000_20260919150000.asf"
    res1 = parse_dahua_filename(f1)
    assert res1 is not None
    assert res1.channel == 4
    assert res1.stream_tier == "main"
    assert res1.is_substream is False

    f2 = "channel9_extra1_20260919080000_20260919090000.dav"
    res2 = parse_dahua_filename(f2)
    assert res2 is not None
    assert res2.channel == 9
    assert res2.stream_tier == "extra1"
    assert res2.is_substream is True

    # Generic test video fallback
    f3 = "test2.mp4"
    res3 = parse_dahua_filename(f3)
    assert res3 is not None
    assert res3.channel == 2
    assert res3.is_substream is True


def test_parse_dahua_filename_unsupported():
    """Tests that non-video or unparseable files return None."""
    assert parse_dahua_filename("readme.txt") is None
    assert parse_dahua_filename("image.jpg") is None
    assert parse_dahua_filename("surveillance_notes.pdf") is None


def test_discover_dahua_videos_filtering(tmp_path: Path):
    """Tests directory discovery with stream tier and channel filters."""
    # Create mock video files
    (tmp_path / "NVR_ch01_extra1_20260919100000_20260919110000.asf").write_text("dummy")
    (tmp_path / "NVR_ch01_main_20260919100000_20260919110000.asf").write_text("dummy")
    (tmp_path / "NVR_ch02_extra1_20260919100000_20260919110000.asf").write_text("dummy")
    (tmp_path / "NVR_ch03_extra1_20260919100000_20260919110000.asf").write_text("dummy")
    (tmp_path / "unrelated_file.txt").write_text("dummy")

    # 1. Filter sub-stream only (default)
    sub_only = discover_dahua_videos(tmp_path, stream_tier="extra1")
    assert len(sub_only) == 3
    assert all(v.is_substream for v in sub_only)

    # 2. Filter channel 1 only
    ch1_only = discover_dahua_videos(tmp_path, channels=[1], stream_tier="any")
    assert len(ch1_only) == 2
    assert all(v.channel == 1 for v in ch1_only)

    # 3. Filter channel 2 sub-stream
    ch2_sub = discover_dahua_videos(tmp_path, channels=[2], stream_tier="extra1")
    assert len(ch2_sub) == 1
    assert ch2_sub[0].channel == 2


def test_cluster_channel_embeddings_condensation():
    """Tests MiniBatchKMeans clustering and unit L2 normalization."""
    # Create 50 random 576-dim vectors
    rng = np.random.RandomState(42)
    raw_vecs = rng.randn(50, 576).astype(np.float32)
    # L2 normalize inputs
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    embeddings = raw_vecs / norms

    metadata = [{"channel": 1, "clip_idx": i} for i in range(50)]

    # 1. Condense 50 vectors to 10 centroids
    centroids, metas = cluster_channel_embeddings(
        embeddings=embeddings,
        metadata_list=metadata,
        n_clusters=10,
        batch_size=20,
    )
    assert centroids.shape == (10, 576)
    assert len(metas) == 10
    # Verify all centroids are strictly unit L2 normalized
    c_norms = np.linalg.norm(centroids, axis=1)
    assert np.allclose(c_norms, 1.0, atol=1e-4)
    assert all(m["is_centroid"] is True for m in metas)

    # 2. Test small sample fallback (N <= n_clusters)
    small_vecs = embeddings[:5]
    s_centroids, s_metas = cluster_channel_embeddings(
        embeddings=small_vecs,
        metadata_list=metadata[:5],
        n_clusters=10,
    )
    assert s_centroids.shape == (5, 576)
    assert len(s_metas) == 5


def test_index_channel_centroids_dual_upsert():
    """Tests dual upsert into Central Qdrant and Edge Qdrant."""
    centroids = np.zeros((3, 576), dtype=np.float32)
    centroids[:, 0] = 1.0  # Unit vectors
    metas = [{"channel": 1, "is_centroid": True, "cluster_id": i} for i in range(3)]

    mock_edge_detector = MagicMock()
    mock_edge_detector.seed_baseline.return_value = 3

    with patch("scripts.embed_and_index_baseline.index_baseline_vectors", return_value=3) as mock_cloud:
        res = index_channel_centroids(
            channel=1,
            centroids=centroids,
            metadata_list=metas,
            upload_cloud=True,
            upload_edge=True,
            edge_detector=mock_edge_detector,
            collection_name="test_baseline",
        )

        assert res["cloud_points"] == 3
        assert res["edge_points"] == 3
        mock_cloud.assert_called_once()
        mock_edge_detector.seed_baseline.assert_called_once()


def test_run_baseline_pipeline_dry_run():
    """Tests end-to-end baseline pipeline execution in dry-run mode on test videos."""
    test_dir = Path(__file__).resolve().parent.parent / "data" / "tests"
    if not test_dir.exists():
        pytest.skip("data/tests not present")

    res = run_baseline_pipeline(
        video_dir=test_dir,
        clusters_per_camera=3,
        channels=[1, 2],
        dry_run=True,
        max_windows_per_video=2,
    )

    assert res["status"] == "success"
    assert res["channels_processed"] == 2
    assert res["total_windows"] > 0
    assert res["total_centroids"] > 0
    assert 1 in res["channel_results"]
    assert 2 in res["channel_results"]


def test_clear_vector_databases(tmp_path: Path):
    """Tests clearing vector databases for edge and cloud."""
    from scripts.embed_and_index_baseline import clear_vector_databases

    edge_dir = tmp_path / "qdrant_edge"
    edge_dir.mkdir()
    (edge_dir / "test_file.bin").write_text("dummy")

    with patch("qdrant_client.QdrantClient") as mock_qclient:
        mock_instance = mock_qclient.return_value
        mock_instance.get_collections.return_value.collections = [
            MagicMock(name="test_baseline")
        ]

        res = clear_vector_databases(
            clear_cloud=True,
            clear_edge=True,
            collection_name="test_baseline",
            qdrant_edge_path=edge_dir,
            embedding_dim=576,
        )

        assert res["cloud_cleared"] is True
        assert res["edge_cleared"] is True
        assert not (edge_dir / "test_file.bin").exists()
