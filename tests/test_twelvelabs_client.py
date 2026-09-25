"""Tests for Twelve Labs client wrapper and error handling."""

from unittest.mock import MagicMock, patch
import pytest

from backend import twelvelabs_client
from backend.config import config


def test_twelvelabs_enabled_check():
    original_key = config.twelve_labs_api_key
    try:
        config.twelve_labs_api_key = ""
        assert twelvelabs_client.is_enabled() is False

        config.twelve_labs_api_key = "tlk_test_12345"
        assert twelvelabs_client.is_enabled() is True
    finally:
        config.twelve_labs_api_key = original_key


def test_get_client_raises_when_disabled():
    original_key = config.twelve_labs_api_key
    twelvelabs_client._client = None
    try:
        config.twelve_labs_api_key = ""
        with pytest.raises(RuntimeError, match="TWELVE_LABS_API_KEY is not configured"):
            twelvelabs_client.get_client()
    finally:
        config.twelve_labs_api_key = original_key


def test_search_videos_mocked():
    mock_client = MagicMock()
    mock_clip = MagicMock()
    mock_clip.video_id = "vid_123"
    mock_clip.score = 88.5
    mock_clip.start = 12.0
    mock_clip.end = 22.0
    mock_clip.confidence = "high"
    mock_clip.module_type = "visual"

    mock_group = MagicMock()
    mock_group.clips = [mock_clip]

    mock_search_resp = MagicMock()
    mock_search_resp.data = [mock_group]

    mock_client.search.query.return_value = mock_search_resp

    with patch.object(twelvelabs_client, "get_client", return_value=mock_client), \
         patch.object(twelvelabs_client, "get_marengo_index_id", return_value="idx_123"):

        results = twelvelabs_client.search_videos(query="person at driveway", max_clips=5)
        assert len(results) == 1
        assert results[0].video_id == "vid_123"
        assert results[0].score == 88.5
        assert results[0].start == 12.0


def test_search_videos_syncpager_mocked():
    """Tests Twelve Labs SDK v1.3+ SyncPager direct SearchItem iteration."""
    mock_client = MagicMock()
    mock_item = MagicMock(spec=["video_id", "score", "start", "end", "confidence"])
    mock_item.video_id = "idx_asset_777"
    mock_item.score = 0.94
    mock_item.start = 3.5
    mock_item.end = 9.0
    mock_item.confidence = "high"
    mock_item.clips = None

    # Simulates SyncPager iterator (without .data attribute)
    mock_pager = [mock_item]
    mock_client.search.query.return_value = mock_pager

    with patch.object(twelvelabs_client, "get_client", return_value=mock_client), \
         patch.object(twelvelabs_client, "get_marengo_index_id", return_value="idx_123"):

        results = twelvelabs_client.search_videos(query="delivery courier", max_clips=5)
        assert len(results) == 1
        assert results[0].video_id == "idx_asset_777"
        assert results[0].score == 0.94
        assert results[0].start == 3.5


def test_analyze_video_pegasus_stream():
    """Tests Pegasus 1.5 analyze_stream handling matching official Twelve Labs starter code."""
    mock_client = MagicMock(spec=["analyze_stream", "search", "embed"])
    chunk1 = MagicMock(event_type="text_generation", text="A delivery courier ")
    chunk2 = MagicMock(event_type="text_generation", text="placed a cardboard package on the porch.")
    chunk3 = MagicMock(event_type="other_event", text="ignore this")
    mock_client.analyze_stream.return_value = [chunk1, chunk2, chunk3]

    with patch.object(twelvelabs_client, "get_client", return_value=mock_client):
        res = twelvelabs_client.analyze_video(video_id="ast_12345", prompt="Summarize this incident.")
        assert "cardboard package" in res.text
        assert res.video_id == "ast_12345"
        assert res.latency_ms >= 0.0
        mock_client.analyze_stream.assert_called_once()


def test_analyze_video_direct_fallback():
    """Tests Pegasus 1.5 client.analyze direct call when analyze_stream is absent."""
    mock_client = MagicMock(spec=["analyze", "search", "embed"])
    mock_resp = MagicMock()
    mock_resp.data = "Person observed walking near front gate."
    mock_client.analyze.return_value = mock_resp

    with patch.object(twelvelabs_client, "get_client", return_value=mock_client):
        res = twelvelabs_client.analyze_video(video_id="ast_67890", prompt="Describe incident.")
        assert "front gate" in res.text
        assert res.video_id == "ast_67890"


def test_analyze_video_legacy_generate_fallback():
    """Tests backward compatibility fallback to client.generate.text."""
    mock_client = MagicMock(spec=["generate", "search", "embed"])
    mock_resp = MagicMock()
    mock_resp.data = "A delivery courier placed a cardboard package by the front porch."
    mock_client.generate.text.return_value = mock_resp

    with patch.object(twelvelabs_client, "get_client", return_value=mock_client):
        res = twelvelabs_client.analyze_video(video_id="vid_abc", prompt="Summarize this incident.")
        assert "package" in res.text
        assert res.video_id == "vid_abc"


def test_upload_video_direct_asset_success(tmp_path):
    """Tests direct asset upload and status polling matching official Twelve Labs flow."""
    fake_video = tmp_path / "test_clip.mp4"
    fake_video.write_bytes(b"dummy mp4 video bytes")

    mock_client = MagicMock()
    mock_client.assets.list.return_value = []
    mock_asset = MagicMock(id="ast_test_999")
    mock_client.assets.create.return_value = mock_asset
    mock_client.indexes.indexed_assets.create.return_value.id = "ast_test_999"

    # Polling returns pending then ready
    asset_pending = MagicMock(status="pending")
    asset_ready = MagicMock(status="ready")
    mock_client.assets.retrieve.side_effect = [asset_pending, asset_ready]

    with patch.object(twelvelabs_client, "get_client", return_value=mock_client), \
         patch.object(twelvelabs_client, "_load_asset_cache", return_value={}):
        res = twelvelabs_client.upload_video(file_path=fake_video)
        assert res["pegasus_video_id"] == "ast_test_999"
        assert res["marengo_video_id"] == "ast_test_999"
        mock_client.assets.create.assert_called_once()
        assert mock_client.assets.retrieve.call_count == 2


def test_upload_video_url_asset_success():
    """Tests URL asset upload method."""
    mock_client = MagicMock()
    mock_client.assets.list.return_value = []
    mock_asset = MagicMock(id="ast_url_555")
    mock_client.assets.create.return_value = mock_asset
    mock_client.assets.retrieve.return_value = MagicMock(status="ready")

    with patch.object(twelvelabs_client, "get_client", return_value=mock_client), \
         patch.object(twelvelabs_client, "_load_asset_cache", return_value={}):
        res = twelvelabs_client.upload_video(file_path="https://example.com/stream/camera1.mp4")
        assert res["pegasus_video_id"] == "ast_url_555"
        mock_client.assets.create.assert_called_once_with(
            method="url",
            url="https://example.com/stream/camera1.mp4",
            filename="camera1.mp4",
        )


def test_upload_video_deduplication_reused(tmp_path):
    """Tests that existing ready assets are reused without creating duplicate assets."""
    fake_video = tmp_path / "test_existing.mp4"
    fake_video.write_bytes(b"dummy bytes")

    mock_client = MagicMock()
    mock_existing_asset = MagicMock()
    mock_existing_asset.id = "ast_cached_123"
    mock_existing_asset.filename = "test_existing.mp4"
    mock_existing_asset.status = "ready"
    mock_client.assets.list.return_value = [mock_existing_asset]

    with patch.object(twelvelabs_client, "get_client", return_value=mock_client), \
         patch.object(twelvelabs_client, "_load_asset_cache", return_value={}):
        res = twelvelabs_client.upload_video(file_path=fake_video)
        assert res["pegasus_video_id"] == "ast_cached_123"
        # Assets create must NOT be called because it was deduplicated
        mock_client.assets.create.assert_not_called()


def test_upload_video_asset_failure(tmp_path):
    """Tests asset processing failure status exception."""
    fake_video = tmp_path / "corrupt_clip.mp4"
    fake_video.write_bytes(b"corrupt bytes")

    mock_client = MagicMock()
    mock_client.assets.list.return_value = []
    mock_asset = MagicMock(id="ast_fail_111")
    mock_client.assets.create.return_value = mock_asset
    mock_client.assets.retrieve.return_value = MagicMock(status="failed")

    with patch.object(twelvelabs_client, "get_client", return_value=mock_client), \
         patch.object(twelvelabs_client, "_load_asset_cache", return_value={}):
        with pytest.raises(RuntimeError, match="Twelve Labs asset processing failed"):
            twelvelabs_client.upload_video(file_path=fake_video)

