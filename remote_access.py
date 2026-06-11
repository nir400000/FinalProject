"""
WebRTC remote video for the baby monitor.

The Pi connects outbound to the public signaling server (no router port forwarding).
Uses aiortc to publish the camera as a WebRTC video track.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Optional

import cv2
import numpy as np
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.sdp import candidate_from_sdp
from av import VideoFrame

from device_registry import get_device_config
from frame_buffer import get_frame_copy

logger = logging.getLogger(__name__)

RTC_CONFIGURATION = RTCConfiguration(
    iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
)


class CameraStreamTrack(VideoStreamTrack):
    kind = "video"

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame_bgr = get_frame_copy()
        if frame_bgr is None:
            frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        await asyncio.sleep(0.05)
        return video_frame


class RemoteAccessService:
    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._pc: Optional[RTCPeerConnection] = None
        self._ws = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        cfg = get_device_config()
        if not cfg.get("signaling_url"):
            logger.warning(
                "Remote access disabled: set BABYMONITOR_SIGNALING_URL "
                "or signaling_url in device_config.json"
            )
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="remote-webrtc")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception:
            logger.exception("Remote access loop stopped")
        finally:
            self._loop.close()

    async def _shutdown(self) -> None:
        await self._reset_peer()
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def _reset_peer(self) -> None:
        if self._pc:
            try:
                await self._pc.close()
            except Exception:
                logger.exception("Failed to close peer connection")
            self._pc = None

    async def _main(self) -> None:
        import websockets

        cfg = get_device_config()
        url = cfg["signaling_url"]
        device_id = cfg["device_id"]
        token = cfg["access_token"]

        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    self._ws = ws
                    await ws.send(
                        json.dumps(
                            {
                                "type": "register",
                                "role": "monitor",
                                "device_id": device_id,
                                "token": token,
                            }
                        )
                    )
                    logger.info("Connected to signaling server as monitor %s", device_id)
                    await self._listen(ws)
            except Exception:
                logger.exception("Signaling connection failed; retrying in 10s")
                await asyncio.sleep(10)

    async def _listen(self, ws) -> None:
        async for raw in ws:
            if self._stop.is_set():
                break
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Ignoring invalid signaling JSON")
                continue
            if message.get("type") == "signal":
                await self._handle_signal(ws, message.get("payload", {}))

    async def _ensure_peer(self, ws) -> RTCPeerConnection:
        if self._pc:
            return self._pc
        pc = RTCPeerConnection(configuration=RTC_CONFIGURATION)
        pc.addTrack(CameraStreamTrack())

        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate is None:
                return
            await ws.send(
                json.dumps(
                    {
                        "type": "signal",
                        "payload": {
                            "kind": "ice",
                            "candidate": candidate.candidate,
                            "sdpMid": candidate.sdpMid,
                            "sdpMLineIndex": candidate.sdpMLineIndex,
                        },
                    }
                )
            )

        self._pc = pc
        return pc

    async def _handle_signal(self, ws, payload: dict) -> None:
        kind = payload.get("kind")
        try:
            if kind == "offer":
                await self._reset_peer()
                pc = await self._ensure_peer(ws)
                offer = RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await ws.send(
                    json.dumps(
                        {
                            "type": "signal",
                            "payload": {
                                "kind": "answer",
                                "type": pc.localDescription.type,
                                "sdp": pc.localDescription.sdp,
                            },
                        }
                    )
                )
                logger.info("Sent WebRTC answer to viewer")
            elif kind == "ice" and self._pc:
                candidate_str = payload.get("candidate")
                if not candidate_str:
                    return
                ice = candidate_from_sdp(candidate_str)
                ice.sdpMid = payload.get("sdpMid")
                ice.sdpMLineIndex = payload.get("sdpMLineIndex")
                await self._pc.addIceCandidate(ice)
        except Exception:
            logger.exception("Failed to handle WebRTC signal (%s)", kind)
            await self._reset_peer()


_service = RemoteAccessService()


def start_remote_access_thread() -> None:
    _service.start()


def stop_remote_access() -> None:
    _service.stop()
