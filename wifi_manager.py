"""Wi-Fi scan/connect helpers for Raspberry Pi OS (NetworkManager / nmcli)."""

from __future__ import annotations

import json
import re
import subprocess
from typing import List, Optional, Tuple


def _run(cmd: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def get_wifi_interface() -> str:
    """Return the first Wi-Fi interface managed by NetworkManager."""
    result = _run(["nmcli", "-t", "-f", "DEVICE,TYPE,STATE", "dev"], timeout=15)
    for line in result.stdout.splitlines():
        parts = line.split(":")
        if len(parts) >= 2 and parts[1] == "wifi":
            return parts[0]
    return "wlan0"


def _parse_terse_fields(line: str, field_count: int) -> Optional[List[str]]:
    """
    Parse nmcli -t lines. Colons inside values are escaped as \\:
    """
    fields: List[str] = []
    current: List[str] = []
    i = 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == ":":
            current.append(":")
            i += 2
            continue
        if line[i] == ":":
            fields.append("".join(current))
            current = []
            i += 1
            if len(fields) == field_count - 1:
                fields.append(line[i:])
                return fields
            continue
        current.append(line[i])
        i += 1
    if current or fields:
        fields.append("".join(current))
    return fields if len(fields) == field_count else None


def scan_networks(max_networks: int = 25) -> List[dict]:
    """Return visible Wi-Fi networks sorted by signal strength."""
    try:
        _run(["nmcli", "dev", "wifi", "rescan"], timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    result = _run(
        ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi", "list"],
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Wi-Fi scan failed")

    networks: List[dict] = []
    seen = set()
    for line in result.stdout.splitlines():
        if not line:
            continue
        parsed = _parse_terse_fields(line, 3)
        if not parsed:
            continue
        ssid, signal_text, security = parsed[0].strip(), parsed[1].strip(), parsed[2].strip()
        if not ssid or ssid in seen:
            continue
        seen.add(ssid)
        try:
            rssi = int(signal_text)
        except ValueError:
            rssi = 0
        secure = bool(security and security != "--")
        networks.append(
            {
                "ssid": ssid,
                "rssi": rssi,
                "secure": secure,
                "security": security if security != "--" else "",
            }
        )

    networks.sort(key=lambda item: item["rssi"], reverse=True)
    return networks[:max_networks]


def _delete_stale_profiles(ssid: str) -> None:
    """Remove broken/auto-created profiles that block a new connection."""
    result = _run(["nmcli", "-t", "-f", "NAME", "connection", "show"], timeout=15)
    for name in result.stdout.splitlines():
        name = name.strip()
        if not name:
            continue
        if name == ssid or name.startswith("baby-monitor-"):
            _run(["nmcli", "connection", "delete", name], timeout=15)


def _key_mgmt_modes(security: str) -> List[str]:
    """Return key-mgmt values to try, in order."""
    sec = security.upper()
    if not sec or sec == "--":
        return []

    modes: List[str] = []
    # WPA3-personal only
    if "WPA3" in sec and "WPA2" not in sec and "WPA1" not in sec:
        modes.append("sae")
    # WPA2 / mixed / generic secured
    if "WPA2" in sec or "WPA1" in sec or "WPA" in sec:
        modes.append("wpa-psk")
    if "WPA3" in sec and "wpa-psk" not in modes:
        modes.append("wpa-psk")
    if "SAE" in sec and "sae" not in modes:
        modes.append("sae")

    # Default for unknown secured networks (most home routers)
    if not modes:
        modes = ["wpa-psk", "sae"]
    return modes


def _connect_with_profile(
    ssid: str,
    password: str,
    key_mgmt: str,
    ifname: str,
) -> Tuple[bool, str]:
    con_name = re.sub(r"[^a-zA-Z0-9._-]", "_", f"baby-monitor-{ssid}")[:32]

    _delete_stale_profiles(ssid)

    add_cmd = [
        "nmcli",
        "connection",
        "add",
        "type",
        "wifi",
        "con-name",
        con_name,
        "ifname",
        ifname,
        "ssid",
        ssid,
        "wifi-sec.key-mgmt",
        key_mgmt,
        "wifi-sec.psk",
        password,
    ]
    add_result = _run(add_cmd, timeout=30)
    if add_result.returncode != 0:
        return False, (add_result.stderr or add_result.stdout or "").strip()

    up_result = _run(["nmcli", "-w", "90", "connection", "up", con_name], timeout=120)
    output = (up_result.stderr or up_result.stdout or add_result.stdout or "").strip()
    if up_result.returncode == 0:
        return True, output

    _run(["nmcli", "connection", "delete", con_name], timeout=15)
    return False, output


def connect_network(
    ssid: str,
    password: str | None,
    security: str = "",
) -> Tuple[bool, str]:
    """Connect to a Wi-Fi network. Returns (success, message)."""
    ifname = get_wifi_interface()
    pwd = password or ""

    _delete_stale_profiles(ssid)

    # Open network
    if not pwd:
        cmd = ["nmcli", "-w", "90", "dev", "wifi", "connect", ssid, "ifname", ifname]
        result = _run(cmd, timeout=120)
        output = (result.stderr or result.stdout or "").strip()
        return result.returncode == 0, output

    last_error = ""
    modes = _key_mgmt_modes(security)

    for key_mgmt in modes:
        ok, msg = _connect_with_profile(ssid, pwd, key_mgmt, ifname)
        if ok:
            return True, msg
        last_error = msg

    # Last resort: inline connect with explicit key-mgmt (older nmcli style)
    for key_mgmt in modes or ["wpa-psk", "sae"]:
        cmd = [
            "nmcli",
            "-w",
            "90",
            "dev",
            "wifi",
            "connect",
            ssid,
            "ifname",
            ifname,
            "password",
            pwd,
            "wifi-sec.key-mgmt",
            key_mgmt,
        ]
        result = _run(cmd, timeout=120)
        output = (result.stderr or result.stdout or "").strip()
        if result.returncode == 0:
            return True, output
        last_error = output
        _delete_stale_profiles(ssid)

    return False, last_error or "Connection failed"


def _extract_ipv4(text: str) -> str:
    match = re.search(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b", text)
    return match.group(1) if match else ""


def get_primary_ip() -> str:
    """Best-effort IPv4 address for the wireless interface."""
    ifname = get_wifi_interface()
    result = _run(["nmcli", "-t", "-f", "IP4.ADDRESS", "dev", "show", ifname], timeout=10)
    for line in result.stdout.splitlines():
        if "IP4.ADDRESS" not in line:
            continue
        ip = _extract_ipv4(line)
        if ip and not ip.startswith("127."):
            return ip

    result = _run(["hostname", "-I"], timeout=10)
    for token in result.stdout.split():
        ip = _extract_ipv4(token)
        if ip and not ip.startswith("127."):
            return ip
    return ""


def networks_to_json(networks: List[dict]) -> str:
    return json.dumps({"networks": networks})
