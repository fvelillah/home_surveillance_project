"""Edge device and camera registry for tracking connected nodes and health."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import config


@dataclass
class EdgeDeviceRecord:
    """State and operational metrics for an edge device or camera channel."""
    device_id: str
    name: str
    location: str
    status: str = "online"
    baseline_version: str = "v-init"
    threshold: float = config.cloud_anomaly_threshold
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    escalation_count: int = 0
    confirmed_count: int = 0

    def record_escalation(self, is_confirmed: bool):
        self.escalation_count += 1
        if is_confirmed:
            self.confirmed_count += 1
        self.last_seen = time.time()

    def stats(self) -> Dict[str, Any]:
        confirmation_rate = (self.confirmed_count / self.escalation_count) if self.escalation_count > 0 else 0.0
        return {
            "device_id": self.device_id,
            "name": self.name,
            "location": self.location,
            "status": self.status,
            "baseline_version": self.baseline_version,
            "threshold": self.threshold,
            "escalation_count": self.escalation_count,
            "confirmed_count": self.confirmed_count,
            "confirmation_rate": round(confirmation_rate, 4),
            "last_seen_s_ago": round(time.time() - self.last_seen, 2),
        }


class EdgeRegistry:
    """Registry maintaining edge device states and 9-channel camera inventory."""

    def __init__(self):
        self._devices: Dict[str, EdgeDeviceRecord] = {}
        self._init_default_cameras()

    def _init_default_cameras(self):
        """Initializes default 9 Dahua camera records from configuration."""
        cameras = config.parse_camera_names()
        for ch, name in cameras.items():
            device_id = f"cam-{ch}"
            self._devices[device_id] = EdgeDeviceRecord(
                device_id=device_id,
                name=f"Channel {ch}: {name}",
                location=name,
                threshold=config.cloud_anomaly_threshold,
            )

    def register(
        self,
        name: str,
        location: str,
        device_id: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> EdgeDeviceRecord:
        dev_id = device_id or f"edge-{str(uuid.uuid4())[:8]}"
        record = EdgeDeviceRecord(
            device_id=dev_id,
            name=name,
            location=location,
            threshold=threshold if threshold is not None else config.cloud_anomaly_threshold,
        )
        self._devices[dev_id] = record
        return record

    def get(self, device_id: str) -> Optional[EdgeDeviceRecord]:
        return self._devices.get(device_id)

    def list_all(self) -> List[EdgeDeviceRecord]:
        return list(self._devices.values())

    def update_heartbeat(self, device_id: str):
        if device_id in self._devices:
            self._devices[device_id].last_seen = time.time()
            self._devices[device_id].status = "online"

    def set_baseline_version(self, device_id: str, version: str):
        if device_id in self._devices:
            self._devices[device_id].baseline_version = version

    def set_threshold(self, device_id: str, threshold: float):
        if device_id in self._devices:
            self._devices[device_id].threshold = threshold


# Global registry singleton
registry = EdgeRegistry()
