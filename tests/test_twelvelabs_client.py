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


def test_analyze_video_mocked():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.data = "A delivery courier placed a cardboard package by the front porch."
    mock_client.generate.text.return_value = mock_resp

    with patch.object(twelvelabs_client, "get_client", return_value=mock_client):
        res = twelvelabs_client.analyze_video(video_id="vid_abc", prompt="Summarize this incident.")
        assert "package" in res.text
        assert res.video_id == "vid_abc"
