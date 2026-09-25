"""Tests for standalone model server embedding extraction."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from model_server import app, extract_embedding_from_path, load_local_model


def _create_synthetic_mp4(filename: str, num_frames: int = 16, width: int = 64, height: int = 64):
    """Generates a small test MP4 video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, 10.0, (width, height))
    for i in range(num_frames):
        # Color gradient frame
        frame = np.full((height, width, 3), (i * 15) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()


def test_extract_embedding_from_path():
    load_local_model("cpu")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        _create_synthetic_mp4(tmp_path, num_frames=16)
        emb = extract_embedding_from_path(tmp_path)

        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 1
        assert len(emb) in (512, 576, 1280)
        # Verify L2 norm is ~1.0
        norm = np.linalg.norm(emb)
        assert pytest.approx(norm, 1e-4) == 1.0
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_model_server_api_endpoints():
    client = TestClient(app)

    # 1. Health check
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

    # 2. Upload and embed video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        _create_synthetic_mp4(tmp_path, num_frames=16)
        with open(tmp_path, "rb") as f:
            resp_embed = client.post(
                "/embed",
                files={"file": ("test.mp4", f, "video/mp4")},
            )
        assert resp_embed.status_code == 200
        embed_data = resp_embed.json()
        assert "embedding" in embed_data
        assert embed_data["dim"] in (512, 576, 1280)
        assert len(embed_data["embedding"]) == embed_data["dim"]
        assert embed_data["latency_ms"] >= 0.0

        # 3. Batch embed
        with open(tmp_path, "rb") as f1, open(tmp_path, "rb") as f2:
            resp_batch = client.post(
                "/batch-embed",
                files=[
                    ("files", ("test1.mp4", f1, "video/mp4")),
                    ("files", ("test2.mp4", f2, "video/mp4")),
                ],
            )
        assert resp_batch.status_code == 200
        batch_data = resp_batch.json()
        assert batch_data["count"] == 2
        assert len(batch_data["embeddings"]) == 2
    finally:
        Path(tmp_path).unlink(missing_ok=True)
