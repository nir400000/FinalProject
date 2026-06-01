"""Wi-Fi scan/connect helpers for Raspberry Pi OS (NetworkManager / nmcli)."""

from __future__ import annotations

import json
import subprocess
from typing import List, Tuple


def scan_networks(max_networks: int = 25) -> List[dict]:
    """Return visible Wi-Fi networks sorted by signal strength."""
    try:
        subprocess.run(
            ["nmcli", "dev", "wifi", "rescan"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    result = subprocess.run(
        ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi", "list"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Wi-Fi scan failed")

    networks: List[dict] = []
    seen = set()
    for line in result.stdout.splitlines():
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 3:
            continue
        ssid = parts[0].strip()
        if not ssid or ssid in seen:
            continue
        seen.add(ssid)
        try:
            rssi = int(parts[1])
        except ValueError:
            rssi = 0
        security = parts[2].strip()
        secure = bool(security and security != "--")
        networks.append({"ssid": ssid, "rssi": rssi, "secure": secure})

    networks.sort(key=lambda item: item["rssi"], reverse=True)
    return networks[:max_networks]


def connect_network(ssid: str, password: str | None) -> Tuple[bool, str]:
    """Connect to a Wi-Fi network. Returns (success, message)."""
    cmd = ["nmcli", "-w", "90", "dev", "wifi", "connect", ssid]
    if password:
        cmd.extend(["password", password])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = (result.stderr or result.stdout or "").strip()
    return result.returncode == 0, output


def get_primary_ip() -> str:
    """Best-effort IPv4 address for the wireless interface."""
    for dev in ("wlan0", "wlP7p1s0", "wlx000000000000"):
        result = subprocess.run(
            ["nmcli", "-t", "-f", "IP4.ADDRESS", "dev", "show", dev],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        for line in result.stdout.splitlines():
            if "IP4.ADDRESS" in line:
                value = line.split(":")[-1].strip()
                if value:
                    return value.split("/")[0]

    result = subprocess.run(
        ["hostname", "-I"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    for token in result.stdout.split():
        if "." in token:
            return token
    return ""


def networks_to_json(networks: List[dict]) -> str:
    return json.dumps({"networks": networks})
