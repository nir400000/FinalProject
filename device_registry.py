"""Persistent device identity and remote-access credentials for the monitor."""

from __future__ import annotations

import json
import os
import secrets
import uuid
from pathlib import Path
from typing import Dict

CONFIG_PATH = Path(__file__).resolve().parent / "device_config.json"
DEFAULT_SIGNALING_URL = os.environ.get("BABYMONITOR_SIGNALING_URL", "").strip()


def _load() -> Dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _save(data: Dict) -> None:
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_device_config() -> Dict:
    data = _load()
    changed = False
    if not data.get("device_id"):
        data["device_id"] = str(uuid.uuid4())
        changed = True
    if not data.get("access_token"):
        data["access_token"] = secrets.token_urlsafe(32)
        changed = True
    if "signaling_url" not in data:
        data["signaling_url"] = DEFAULT_SIGNALING_URL
        changed = True
    if changed:
        _save(data)
    return data


def get_remote_info_json() -> str:
    cfg = get_device_config()
    return json.dumps(
        {
            "device_id": cfg["device_id"],
            "access_token": cfg["access_token"],
            "signaling_url": cfg.get("signaling_url", ""),
        }
    )


def set_signaling_url(url: str) -> None:
    data = get_device_config()
    data["signaling_url"] = url.strip()
    _save(data)
