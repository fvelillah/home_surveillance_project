"""Automated unit tests for Copilot CLI tool (scripts/copilot_cli.py)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.copilot import copilot_engine
from backend.incidents import incident_engine
from scripts.copilot_cli import (
    auto_seed_surveillance_data,
    main,
)


@pytest.fixture(autouse=True)
def clean_copilot_and_incidents():
    """Cleans incident engine and copilot sessions before each test."""
    incident_engine.clear()
    copilot_engine.clear_all()
    yield
    incident_engine.clear()
    copilot_engine.clear_all()


@pytest.mark.asyncio
async def test_auto_seed_surveillance_data_empty_dir(tmp_path: Path):
    """Verifies seeding simulated incidents when directory is empty."""
    count = await auto_seed_surveillance_data(tmp_path, mock_mode=True)
    assert count >= 1
    assert len(incident_engine.list_incidents()) >= 1


def test_cli_main_single_query(tmp_path: Path):
    """Verifies single query CLI execution with JSON output."""
    out_file = tmp_path / "copilot_resp.json"
    cli_args = [
        "copilot_cli.py",
        "--dir", str(tmp_path),
        "--query", "Who was at the front door today?",
        "--mock",
        "--json",
        "-o", str(out_file),
    ]

    with patch("sys.argv", cli_args):
        code = main()
        assert code == 0

    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "response" in data
    assert "session_id" in data
    assert "evidence" in data


def test_cli_main_digest_generation(tmp_path: Path):
    """Verifies daily digest generation via CLI."""
    out_file = tmp_path / "daily_digest.json"
    cli_args = [
        "copilot_cli.py",
        "--dir", str(tmp_path),
        "--digest",
        "--mock",
        "--json",
        "-o", str(out_file),
    ]

    with patch("sys.argv", cli_args):
        code = main()
        assert code == 0

    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "digest_id" in data
    assert "threat_level" in data
    assert "executive_summary" in data
    assert "channel_summaries" in data
    assert len(data["channel_summaries"]) == 9
