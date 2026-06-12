#!/usr/bin/env python3
"""
Minimal WebRTC signaling relay (deploy on a public server).

Both the Raspberry Pi monitor and the phone app connect outbound via WebSocket.
No port forwarding is required on the home router.

Run:
  pip install websockets
  python signaling_server.py --host 0.0.0.0 --port 8765

Use wss:// behind HTTPS reverse proxy in production (nginx/caddy).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
from typing import Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

monitors: Dict[str, object] = {}
viewers: Dict[str, object] = {}
tokens: Dict[str, str] = {}


async def send_json(ws, payload: dict) -> bool:
    try:
        await ws.send(json.dumps(payload))
        return True
    except ConnectionClosed:
        return False


def _remove_stale(role: str, device_id: str, ws) -> None:
    if role == "monitor" and monitors.get(device_id) is ws:
        monitors.pop(device_id, None)
        tokens.pop(device_id, None)
    elif role == "viewer" and viewers.get(device_id) is ws:
        viewers.pop(device_id, None)


async def relay_to_peer(device_id: str, from_role: str, payload: dict) -> None:
    to_role = "viewer" if from_role == "monitor" else "monitor"
    target = viewers.get(device_id) if from_role == "monitor" else monitors.get(device_id)
    if target is None:
        logger.info("No %s connected for device %s; dropping signal", to_role, device_id)
        return
    ok = await send_json(target, {"type": "signal", "payload": payload})
    if not ok:
        logger.info("Stale %s connection for device %s; removing", to_role, device_id)
        _remove_stale(to_role, device_id, target)


async def handler(ws) -> None:
    device_id: Optional[str] = None
    role: Optional[str] = None
    remote = getattr(ws, "remote_address", None)
    try:
        async for raw in ws:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await send_json(ws, {"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = message.get("type")
            if msg_type == "register":
                device_id = str(message.get("device_id", "")).strip()
                token = str(message.get("token", "")).strip()
                role = str(message.get("role", "")).strip()

                if not device_id or not token or role not in ("monitor", "viewer"):
                    await send_json(ws, {"type": "error", "message": "Invalid register payload"})
                    continue

                if role == "monitor":
                    old = monitors.get(device_id)
                    if old is not None and old is not ws:
                        with contextlib.suppress(Exception):
                            await old.close(1000, "replaced")
                    monitors[device_id] = ws
                    tokens[device_id] = token
                else:
                    if tokens.get(device_id) != token:
                        await send_json(ws, {"type": "error", "message": "Invalid token"})
                        await ws.close()
                        return
                    old = viewers.get(device_id)
                    if old is not None and old is not ws:
                        with contextlib.suppress(Exception):
                            await old.close(1000, "replaced")
                    viewers[device_id] = ws

                await send_json(ws, {"type": "registered", "role": role, "device_id": device_id})
                logger.info("Registered %s for device %s (%s)", role, device_id, remote)

                if role == "viewer" and device_id in monitors:
                    await send_json(ws, {"type": "monitor_online"})
                if role == "monitor" and device_id in viewers:
                    await send_json(viewers[device_id], {"type": "monitor_online"})
                continue

            if msg_type == "signal":
                if not device_id or not role:
                    await send_json(ws, {"type": "error", "message": "Not registered"})
                    continue
                payload = message.get("payload")
                if not isinstance(payload, dict):
                    await send_json(ws, {"type": "error", "message": "Invalid signal payload"})
                    continue
                await relay_to_peer(device_id, role, payload)
                continue

            await send_json(ws, {"type": "error", "message": f"Unknown type: {msg_type}"})
    except ConnectionClosed as exc:
        logger.info(
            "Client disconnected (%s, device=%s, peer=%s): %s",
            role or "unknown",
            device_id or "?",
            remote,
            exc,
        )
    except Exception:
        logger.exception("Signaling handler error (%s, device=%s)", role, device_id)
    finally:
        if device_id and role == "monitor" and monitors.get(device_id) is ws:
            monitors.pop(device_id, None)
            tokens.pop(device_id, None)
        if device_id and role == "viewer" and viewers.get(device_id) is ws:
            viewers.pop(device_id, None)


async def main(host: str, port: int) -> None:
    async with websockets.serve(
        handler,
        host,
        port,
        ping_interval=30,
        ping_timeout=120,
        close_timeout=10,
    ):
        logger.info("Signaling server listening on ws://%s:%s", host, port)
        await asyncio.Future()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port))
