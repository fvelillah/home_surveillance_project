"""Pytest configuration and environment isolation."""

import sys
from pathlib import Path

# Project root directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Remove external ROS paths from sys.path to prevent shadowing project modules (e.g., 'scripts')
sys.path = [p for p in sys.path if not p.startswith("/opt/ros")]
