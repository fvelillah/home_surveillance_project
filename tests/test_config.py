"""Tests for EdgeConfig and environment parsing."""

import os
from edge.config import EdgeConfig


def test_edge_config_defaults():
    cfg = EdgeConfig()
    assert cfg.num_cameras >= 1
    assert cfg.ingest_fps > 0
    assert cfg.clip_duration_s > 0
    assert cfg.clip_overlap_s >= 0
    assert cfg.segment_sample_frames > 0


def test_camera_names_parsing():
    raw = "1:Front Door, 2: Driveway ,3: Back Patio"
    parsed = EdgeConfig._parse_camera_names(raw)
    assert parsed[1] == "Front Door"
    assert parsed[2] == "Driveway"
    assert parsed[3] == "Back Patio"


def test_url_generation():
    cfg = EdgeConfig()
    cfg.nvr_ip = "192.168.1.126"
    cfg.nvr_port = 80
    cfg.nvr_user = "testuser"
    cfg.nvr_password = "testpass"

    sub_url = cfg.get_substream_url(2)
    assert "channel=2" in sub_url
    assert "subtype=1" in sub_url
    assert "testuser:testpass@192.168.1.126" in sub_url

    main_url = cfg.get_mainstream_url(2)
    assert "channel=2" in main_url
    assert "subtype=0" in main_url

    snap_url = cfg.get_snapshot_url(2)
    assert "snapshot.cgi" in snap_url
    assert "channel=2" in snap_url
