"""
BLE Wi-Fi provisioning server for the baby monitor (Raspberry Pi).

The phone connects over Bluetooth LE, triggers a Wi-Fi scan, sends credentials,
and reads back connection status plus the monitor's new IP address.
"""

from __future__ import annotations

import json
import logging
import threading
import time

from provisioning_constants import (
    CHAR_CMD_UUID,
    CHAR_DEVICE_IP_UUID,
    CHAR_REMOTE_INFO_UUID,
    CHAR_STATUS_UUID,
    CHAR_WIFI_CRED_UUID,
    CHAR_WIFI_LIST_UUID,
    CMD_SCAN,
    DEVICE_LOCAL_NAME,
    SERVICE_UUID,
)
from ble_chunked_transfer import split_payload
from ble_pairing_agent import start_headless_pairing_agent
from device_registry import get_remote_info_json
from wifi_manager import connect_network, get_primary_ip, networks_to_json, scan_networks

logger = logging.getLogger(__name__)

SVC_ID = 1
CHR_CMD = 1
CHR_WIFI_LIST = 2
CHR_WIFI_CRED = 3
CHR_STATUS = 4
CHR_DEVICE_IP = 5
CHR_REMOTE_INFO = 6


class ProvisionServer:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.wifi_list = networks_to_json([])
        self.status = json.dumps({"state": "idle", "message": "Ready"})
        self.device_ip = ""
        self.ble = None
        self._chars: dict[int, object] = {}

    def _bytes(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def _notify_bytes(self, chr_id: int, data: list[int]) -> None:
        characteristic = self._chars.get(chr_id)
        if characteristic is None:
            return

        def _send() -> bool:
            try:
                characteristic.set_value(data)
            except Exception:
                logger.exception("Failed to send BLE notification for chr_id=%s", chr_id)
            return False

        try:
            from gi.repository import GLib

            GLib.idle_add(_send)
        except Exception:
            _send()

    def _notify(self, chr_id: int, payload: str) -> None:
        """Push a value update (and BLE notification) to connected phones."""
        self._notify_bytes(chr_id, self._bytes(payload))

    def _notify_chunked(self, chr_id: int, payload: str) -> None:
        """Send large payloads as multiple framed notifications."""
        frames = split_payload(payload)
        for index, frame in enumerate(frames):
            self._notify_bytes(chr_id, list(frame))
            if index < len(frames) - 1:
                time.sleep(0.15)

    def _set_status(self, state: str, message: str) -> None:
        with self.lock:
            self.status = json.dumps({"state": state, "message": message})
        logger.info("Provision status: %s - %s", state, message)
        self._notify(CHR_STATUS, self.status)

    def read_wifi_list(self) -> list[int]:
        with self.lock:
            return self._bytes(self.wifi_list)

    def read_status(self) -> list[int]:
        with self.lock:
            return self._bytes(self.status)

    def read_device_ip(self) -> list[int]:
        with self.lock:
            return self._bytes(self.device_ip)

    def read_remote_info(self) -> list[int]:
        return self._bytes(get_remote_info_json())

    def write_cmd(self, value, options) -> None:
        cmd = bytes(value).decode("utf-8", errors="ignore").strip().upper()
        if cmd == CMD_SCAN:
            threading.Thread(target=self._do_scan, daemon=True).start()

    def write_wifi_cred(self, value, options) -> None:
        threading.Thread(target=self._do_connect, args=(bytes(value),), daemon=True).start()

    def _do_scan(self) -> None:
        self._set_status("scanning", "Scanning Wi-Fi networks...")
        try:
            networks = scan_networks(max_networks=12)
            payload = networks_to_json(networks)
            with self.lock:
                self.wifi_list = payload
            self._notify_chunked(CHR_WIFI_LIST, payload)
            time.sleep(0.2)
            self._set_status("idle", f"Found {len(networks)} network(s)")
        except Exception as exc:
            logger.exception("Wi-Fi scan failed")
            self._set_status("error", str(exc))

    def _do_connect(self, raw: bytes) -> None:
        try:
            data = json.loads(raw.decode("utf-8"))
            ssid = str(data.get("ssid", "")).strip()
            password = str(data.get("password", ""))
            security = str(data.get("security", "")).strip()
        except json.JSONDecodeError:
            self._set_status("error", "Invalid credentials payload")
            return

        if not ssid:
            self._set_status("error", "SSID is required")
            return

        self._set_status("connecting", f"Connecting to {ssid}...")
        ok, message = connect_network(ssid, password, security)
        if not ok:
            self._set_status("error", message or "Connection failed")
            return

        time.sleep(3)
        ip = get_primary_ip()
        with self.lock:
            self.device_ip = ip
        if ip:
            self._notify(CHR_DEVICE_IP, ip)
            self._set_status("connected", f"Connected. IP: {ip}")
        else:
            self._set_status("error", "Connected but IP address was not found")

    def _register_characteristic(self, chr_id: int, **kwargs) -> None:
        self.ble.add_characteristic(srv_id=SVC_ID, chr_id=chr_id, **kwargs)
        self._chars[chr_id] = self.ble.characteristics[-1]

    def start(self) -> None:
        from bluezero import adapter as bluez_adapter
        from bluezero import peripheral

        adapters = list(bluez_adapter.Adapter.available())
        if not adapters:
            raise RuntimeError("No Bluetooth adapter found")

        adapter_address = adapters[0].address

        try:
            start_headless_pairing_agent()
        except Exception:
            logger.exception(
                "Headless pairing agent not started; pairing may require a keyboard"
            )

        self.ble = peripheral.Peripheral(adapter_address, local_name=DEVICE_LOCAL_NAME)
        self.ble.add_service(srv_id=SVC_ID, uuid=SERVICE_UUID, primary=True)

        self._register_characteristic(
            CHR_CMD,
            uuid=CHAR_CMD_UUID,
            value=[],
            notifying=False,
            flags=["write", "write-without-response"],
            write_callback=self.write_cmd,
        )
        self._register_characteristic(
            CHR_WIFI_LIST,
            uuid=CHAR_WIFI_LIST_UUID,
            value=self.read_wifi_list(),
            notifying=True,
            flags=["read", "notify"],
            read_callback=self.read_wifi_list,
        )
        self._register_characteristic(
            CHR_WIFI_CRED,
            uuid=CHAR_WIFI_CRED_UUID,
            value=[],
            notifying=False,
            flags=["write", "write-without-response"],
            write_callback=self.write_wifi_cred,
        )
        self._register_characteristic(
            CHR_STATUS,
            uuid=CHAR_STATUS_UUID,
            value=self.read_status(),
            notifying=True,
            flags=["read", "notify"],
            read_callback=self.read_status,
        )
        self._register_characteristic(
            CHR_DEVICE_IP,
            uuid=CHAR_DEVICE_IP_UUID,
            value=self.read_device_ip(),
            notifying=True,
            flags=["read", "notify"],
            read_callback=self.read_device_ip,
        )
        self._register_characteristic(
            CHR_REMOTE_INFO,
            uuid=CHAR_REMOTE_INFO_UUID,
            value=self.read_remote_info(),
            notifying=True,
            flags=["read", "notify"],
            read_callback=self.read_remote_info,
        )

        logger.info(
            "Starting BLE provisioning as '%s' on %s",
            DEVICE_LOCAL_NAME,
            adapter_address,
        )
        self.ble.publish()


_server = ProvisionServer()


def run_ble_provisioning_server() -> None:
    """Blocking call; run inside a background thread."""
    try:
        _server.start()
    except Exception:
        logger.exception("BLE provisioning server stopped")


def start_ble_provisioning_thread() -> threading.Thread:
    thread = threading.Thread(
        target=run_ble_provisioning_server,
        daemon=True,
        name="ble-provision",
    )
    thread.start()
    return thread
