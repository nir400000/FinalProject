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


def _connection_name(ssid: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", f"baby-monitor-{ssid}")[:32]


def _connect_with_profile(
    ssid: str,
    password: str,
    key_mgmt: str,
    ifname: str,
) -> Tuple[bool, str]:
    """Create a NetworkManager profile with explicit WPA settings, then activate it."""
    con_name = _connection_name(ssid)
    _delete_stale_profiles(ssid)

    # Approach A: one-shot add (works on most recent NetworkManager)
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
        "802-11-wireless-security.key-mgmt",
        key_mgmt,
        "802-11-wireless-security.psk",
        password,
    ]
    add_result = _run(add_cmd, timeout=30)
    if add_result.returncode != 0:
        # Approach B: create profile, then set security with modify
        _run(["nmcli", "connection", "delete", con_name], timeout=15)
        base_result = _run(
            [
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
            ],
            timeout=30,
        )
        if base_result.returncode != 0:
            return False, (base_result.stderr or base_result.stdout or add_result.stderr or "").strip()

        modify_result = _run(
            [
                "nmcli",
                "connection",
                "modify",
                con_name,
                "802-11-wireless-security.key-mgmt",
                key_mgmt,
                "802-11-wireless-security.psk",
                password,
            ],
            timeout=30,
        )
        if modify_result.returncode != 0:
            _run(["nmcli", "connection", "delete", con_name], timeout=15)
            return False, (modify_result.stderr or modify_result.stdout or "").strip()

    up_result = _run(["nmcli", "-w", "90", "connection", "up", con_name], timeout=120)
    output = (up_result.stderr or up_result.stdout or "").strip()
    if up_result.returncode == 0:
        return True, output

    _run(["nmcli", "connection", "delete", con_name], timeout=15)
    return False, output


def _connect_with_password_only(ssid: str, password: str, ifname: str) -> Tuple[bool, str]:
    """
    Let NetworkManager infer security from the last scan.
    Valid args for `dev wifi connect`: SSID, password, ifname only.
    """
    cmd = [
        "nmcli",
        "-w",
        "90",
        "dev",
        "wifi",
        "connect",
        ssid,
        "password",
        password,
        "ifname",
        ifname,
    ]
    result = _run(cmd, timeout=120)
    output = (result.stderr or result.stdout or "").strip()
    return result.returncode == 0, output


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

    # Fallback: infer security from scan (no key-mgmt args on dev wifi connect)
    ok, msg = _connect_with_password_only(ssid, pwd, ifname)
    if ok:
        return True, msg

    return False, last_error or msg or "Connection failed"


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
