"""Persistent monitor settings for internet viewing over the private network."""

from __future__ import annotations

import json
import subprocess
from typing import Dict

from server.paths import CONFIG_PATH


def _load() -> Dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _save(data: Dict) -> None:
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_tailscale_ip() -> str:
    data = _load()
    configured = str(data.get("tailscale_ip", "") or "").strip()
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        pass
    return ""


def get_device_config() -> Dict:
    data = _load()
    changed = False
    tailscale_ip = get_tailscale_ip()
    if tailscale_ip and data.get("tailscale_ip") != tailscale_ip:
        if not str(data.get("tailscale_ip", "") or "").strip():
            data["tailscale_ip"] = tailscale_ip
            changed = True
    if changed:
        _save(data)
    return data


def get_remote_info_json() -> str:
    cfg = get_device_config()
    tailscale_ip = str(cfg.get("tailscale_ip", "") or "").strip() or get_tailscale_ip()
    return json.dumps({"tailscale_ip": tailscale_ip})
