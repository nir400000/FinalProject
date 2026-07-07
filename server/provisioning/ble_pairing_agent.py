"""
Headless BlueZ pairing agent for the baby monitor.

Auto-accepts Bluetooth pairing on the Pi so the user never needs a keyboard
to confirm passkeys or matching codes on the monitor.
"""

from __future__ import annotations

import logging

import dbus
import dbus.service
from dbus.mainloop.glib import DBusGMainLoop

logger = logging.getLogger(__name__)

AGENT_INTERFACE = "org.bluez.Agent1"
AGENT_PATH = "/org/babymonitor/agent"
AGENT_CAPABILITY = "NoInputNoOutput"

_agent_registered = False


class HeadlessPairingAgent(dbus.service.Object):
    """BlueZ agent that accepts pairing without user input on the Pi."""

    @dbus.service.method(AGENT_INTERFACE, in_signature="", out_signature="")
    def Release(self) -> None:
        logger.debug("Pairing agent released")

    @dbus.service.method(AGENT_INTERFACE, in_signature="o", out_signature="s")
    def RequestPinCode(self, device: dbus.ObjectPath) -> str:
        logger.info("Auto-accept PIN pairing for %s", device)
        return "0000"

    @dbus.service.method(AGENT_INTERFACE, in_signature="o", out_signature="u")
    def RequestPasskey(self, device: dbus.ObjectPath) -> int:
        logger.info("Auto-accept passkey pairing for %s", device)
        return dbus.UInt32(0)

    @dbus.service.method(AGENT_INTERFACE, in_signature="ouq", out_signature="")
    def DisplayPasskey(self, device: dbus.ObjectPath, passkey: int, entered: int) -> None:
        logger.info("Display passkey %06d for %s (ignored on headless Pi)", passkey, device)

    @dbus.service.method(AGENT_INTERFACE, in_signature="os", out_signature="")
    def DisplayPinCode(self, device: dbus.ObjectPath, pincode: str) -> None:
        logger.info("Display PIN %s for %s (ignored on headless Pi)", pincode, device)

    @dbus.service.method(AGENT_INTERFACE, in_signature="ou", out_signature="")
    def RequestConfirmation(self, device: dbus.ObjectPath, passkey: int) -> None:
        """Numeric-comparison pairing — confirm on Pi without a screen."""
        logger.info(
            "Auto-confirm pairing code %06d for %s (no keyboard needed)",
            passkey,
            device,
        )

    @dbus.service.method(AGENT_INTERFACE, in_signature="o", out_signature="")
    def RequestAuthorization(self, device: dbus.ObjectPath) -> None:
        logger.info("Auto-authorize device %s", device)

    @dbus.service.method(AGENT_INTERFACE, in_signature="os", out_signature="")
    def AuthorizeService(self, device: dbus.ObjectPath, uuid: str) -> None:
        logger.debug("Authorize service %s for %s", uuid, device)

    @dbus.service.method(AGENT_INTERFACE, in_signature="", out_signature="")
    def Cancel(self) -> None:
        logger.debug("Pairing cancelled")


def start_headless_pairing_agent() -> bool:
    """
    Register a default NoInputNoOutput agent with BlueZ.
    Safe to call more than once.
    """
    global _agent_registered

    DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()

    agent_manager = dbus.Interface(
        bus.get_object("org.bluez", "/org/bluez"),
        "org.bluez.AgentManager1",
    )

    if not _agent_registered:
        try:
            agent_manager.RegisterAgent(AGENT_PATH, AGENT_CAPABILITY)
        except dbus.exceptions.DBusException as exc:
            if "AlreadyExists" not in str(exc):
                raise
        HeadlessPairingAgent(bus, AGENT_PATH)
        _agent_registered = True
        logger.info("Registered headless Bluetooth pairing agent")

    agent_manager.RequestDefaultAgent(AGENT_PATH)
    logger.info("Headless pairing agent is default (Just Works / auto-confirm)")
    return True


def stop_headless_pairing_agent() -> None:
    global _agent_registered
    if not _agent_registered:
        return
    try:
        bus = dbus.SystemBus()
        agent_manager = dbus.Interface(
            bus.get_object("org.bluez", "/org/bluez"),
            "org.bluez.AgentManager1",
        )
        agent_manager.UnregisterAgent(AGENT_PATH)
    except Exception:
        logger.exception("Could not unregister pairing agent")
    _agent_registered = False
