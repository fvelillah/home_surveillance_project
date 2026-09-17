"""Unit tests for Adaptive Streaming Backpressure and Load Shedding Manager."""

import pytest

from backend.streaming import StreamingBackpressureManager


@pytest.fixture
def manager():
    mgr = StreamingBackpressureManager(
        auto_mode=True,
        score_only_latency_ms=500.0,
        passthrough_latency_ms=1500.0,
        critical_threshold=0.25,
    )
    mgr.clear()
    return mgr


def test_streaming_manager_default_normal_level(manager):
    assert manager.current_level == 0
    assert manager.level_name == "NORMAL"
    assert not manager.should_skip_vlm()
    assert not manager.should_skip_baseline_knn()
    assert not manager.should_drop_escalation(0.10)


def test_manual_level_override(manager):
    manager.set_level(level=1, auto_mode=False)
    assert manager.current_level == 1
    assert manager.level_name == "SCORE_ONLY"
    assert manager.should_skip_vlm()
    assert not manager.should_skip_baseline_knn()

    manager.set_level(level=2)
    assert manager.current_level == 2
    assert manager.level_name == "PASSTHROUGH"
    assert manager.should_skip_baseline_knn()

    manager.set_level(level=3)
    assert manager.current_level == 3
    assert manager.level_name == "SHED_LOAD"
    # Should drop sub-critical escalations
    assert manager.should_drop_escalation(0.18) is True
    # Should keep critical escalations
    assert manager.should_drop_escalation(0.35) is False


def test_adaptive_level_transitions(manager):
    manager.auto_mode = True

    # 1. Simulate fast requests -> Level 0
    for _ in range(50):
        manager.enter_request()
        manager.exit_request(latency_ms=50.0)
    assert manager.current_level == 0

    # 2. Simulate latency spike above 500ms -> Level 1 (SCORE_ONLY)
    for _ in range(50):
        manager.enter_request()
        manager.exit_request(latency_ms=600.0)
    assert manager.current_level == 1

    # 3. Simulate extreme latency spike above 1500ms -> Level 2 (PASSTHROUGH)
    for _ in range(50):
        manager.enter_request()
        manager.exit_request(latency_ms=1800.0)
    assert manager.current_level == 2

    # 4. Simulate active queue depth overload -> Level 3 (SHED_LOAD)
    for _ in range(55):
        manager.enter_request()
    assert manager.current_level == 3


def test_status_telemetry(manager):
    manager.enter_request()
    manager.exit_request(latency_ms=120.0)

    status = manager.get_status()
    assert status.total_requests == 1
    assert status.avg_latency_ms == 120.0
    assert status.queue_depth == 0
    assert status.level_name == "NORMAL"
