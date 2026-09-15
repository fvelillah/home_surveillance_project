"""Tests for central baseline indexing and kNN anomaly scoring."""

import numpy as np
import pytest

from backend.anomaly import (
    InMemoryCentralBaselineIndex,
    get_collection_stats,
    index_baseline_vectors,
    score_clip,
)


def test_in_memory_central_baseline_index():
    index = InMemoryCentralBaselineIndex()
    index.clear()

    # Seed baseline with 3 normal orthogonal vectors
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    vec3 = [1.0, 1.0, 0.0]
    meta = [
        {"id": "c1", "camera_id": "cam-1", "scene_id": "driveway"},
        {"id": "c2", "camera_id": "cam-2", "scene_id": "backyard"},
        {"id": "c3", "camera_id": "cam-1", "scene_id": "driveway"},
    ]
    total = index.upsert([vec1, vec2, vec3], meta)
    assert total == 3

    # Query with exact match for vec1
    neighbors = index.search(query=[1.0, 0.0, 0.0], k=2, camera_id="cam-1")
    assert len(neighbors) == 2
    assert neighbors[0].clip_id == "c1"
    assert pytest.approx(neighbors[0].similarity, 1e-4) == 1.0

    # Query with filter for cam-2
    neighbors_cam2 = index.search(query=[0.0, 1.0, 0.0], k=1, camera_id="cam-2")
    assert len(neighbors_cam2) == 1
    assert neighbors_cam2[0].clip_id == "c2"


def test_score_clip_normal_and_anomalous():
    # Setup test vectors in baseline
    base_vec = [1.0, 0.0, 0.0, 0.0]
    index_baseline_vectors(
        vectors=[base_vec, base_vec],
        metadata_list=[{"id": "b1", "camera_id": "cam-1"}, {"id": "b2", "camera_id": "cam-1"}],
    )

    # 1. Normal clip matching baseline
    normal_emb = [1.0, 0.0, 0.0, 0.0]
    result_normal = score_clip(
        embedding=normal_emb,
        k=2,
        threshold=0.15,
        camera_id="cam-1",
    )
    assert pytest.approx(result_normal.anomaly_score, 1e-4) == 0.0
    assert result_normal.is_anomaly is False
    assert len(result_normal.neighbors) >= 1

    # 2. Orthogonal anomalous clip
    anomalous_emb = [0.0, 1.0, 0.0, 0.0]
    result_anom = score_clip(
        embedding=anomalous_emb,
        k=2,
        threshold=0.15,
        camera_id="cam-1",
    )
    assert result_anom.anomaly_score >= 0.8
    assert result_anom.is_anomaly is True


def test_collection_stats():
    stats = get_collection_stats()
    assert "collection" in stats
    assert "points_count" in stats
    assert "storage_backend" in stats
