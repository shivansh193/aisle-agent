"""Ring cameras as a Aisle Agent video source (Ring Partner API, api.amazonvision.com).

    RingClient.devices()                 GET /v1/devices (JSON:API) -> [{id, name, online}]
    RingClient.open_stream(device, cb)   WHEP: POST an SDP offer (video recvonly) to
                                         /v1/devices/{id}/media/streaming/whep/sessions, apply the
                                         answer, hand every decoded frame to `cb`; close() DELETEs the
                                         session URL from the Location header
    verify_signature(key, body, header)  webhook X-Signature: "sha256=" + hex(HMAC-SHA256(key, raw body))
    parse_event(payload)                 motion_detected / button_press / ... -> flat dict

Auth: an access token (RING_ACCESS_TOKEN, e.g. from the Developer Portal Playground, valid
30 min) or a refresh token with client credentials (RING_REFRESH_TOKEN, RING_CLIENT_ID,
RING_CLIENT_SECRET), refreshed at https://oauth.ring.com/oauth/token.

Ring caps a live session at 30 s (battery) or 60 s (wired) and stamps a watermark on the video
(logo top-left, device ID / app name top-right, timestamp bottom-right): keep shelf zones
clear of those corners.
"""
import asyncio
import hashlib
import hmac
import logging
import os
import time
from typing import Callable, List, Optional

import httpx

API = "https://api.amazonvision.com"
OAUTH = "https://oauth.ring.com/oauth/token"
log = logging.getLogger("aisle_agent.ring")


def verify_signature(signing_key: str, raw_body: bytes, received: Optional[str]) -> bool:
    """Ring signs the raw request body; the key is used as UTF-8 bytes, as issued."""
    if not signing_key or not received:
        return False
    expected = hmac.new(signing_key.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, received.removeprefix("sha256="))


def parse_event(payload: dict) -> dict:
    """Flatten a Ring webhook (JSON:API, v1.1). Unknown subtypes are passed through."""
    meta, data = payload.get("meta", {}), payload.get("data", {})
    attrs = data.get("attributes", {})
    return {"request_id": meta.get("request_id"), "account_id": meta.get("account_id"), "event_id": data.get("id"),
            "type": data.get("type"), "sub_type": data.get("subType") or data.get("sub_type"),
            "device_id": attrs.get("source"), "timestamp": attrs.get("timestamp")}


class RingClient:
    def __init__(self, access_token: Optional[str] = None, refresh_token: Optional[str] = None,
                 client_id: Optional[str] = None, client_secret: Optional[str] = None, api: str = API,
                 on_refresh: Optional[Callable] = None):
        self.api, self.on_refresh = api, on_refresh
        self._access = access_token
        self._refresh, self._cid, self._secret = refresh_token, client_id, client_secret
        # an access token alone is used until it fails; with a refresh token it is renewed when due
        self._expires = float("inf") if access_token and not refresh_token else 0.0
        if not (access_token or (refresh_token and client_id and client_secret)):
            raise ValueError("Ring auth needs an access token, or a refresh token with client id and secret")

    @classmethod
    def from_env(cls, access_token: Optional[str] = None) -> "RingClient":
        return cls(access_token or os.environ.get("RING_ACCESS_TOKEN"), os.environ.get("RING_REFRESH_TOKEN"),
                   os.environ.get("RING_CLIENT_ID"), os.environ.get("RING_CLIENT_SECRET"))

    def token(self) -> str:
        if self._access and time.time() < self._expires:
            return self._access
        r = httpx.post(OAUTH, data={"grant_type": "refresh_token", "refresh_token": self._refresh,
                                    "client_id": self._cid, "client_secret": self._secret}, timeout=15)
        r.raise_for_status()
        body = r.json()
        self._access = body["access_token"]
        self._expires = time.time() + body.get("expires_in", 3600) - 60
        if body.get("refresh_token"):
            self._refresh = body["refresh_token"]
        if self.on_refresh:
            self.on_refresh(self._access, self._refresh, self._expires)
        return self._access

    def _headers(self, **extra):
        return {"Authorization": f"Bearer {self.token()}", **extra}

    def devices(self) -> List[dict]:
        r = httpx.get(f"{self.api}/v1/devices", headers=self._headers(), timeout=15)
        r.raise_for_status()
        out = []
        for d in r.json().get("data", []):
            a = d.get("attributes", {})
            out.append({"id": d["id"], "name": a.get("name") or a.get("description") or "Ring camera",
                        "online": a.get("online"), "attributes": a})
        return out

    async def open_stream(self, device_id: str, on_frame: Callable, on_end: Optional[Callable] = None,
                          max_seconds: float = 60.0) -> "RingStream":
        s = RingStream(self, device_id, on_frame, on_end, max_seconds)
        await s.start()
        return s


class RingStream:
    """One WHEP live-view session, pulled server-side with aiortc (video only)."""

    def __init__(self, client: RingClient, device_id: str, on_frame: Callable, on_end: Optional[Callable],
                 max_seconds: float):
        self.client, self.device_id = client, device_id
        self.on_frame, self.on_end, self.max_seconds = on_frame, on_end, max_seconds
        self.session_url: Optional[str] = None
        self.frames = 0
        self.started = None
        self.pc = None
        self._task = None
        self._closed = False

    async def start(self):
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer("stun:stun.l.google.com:19302")]))

        @self.pc.on("track")                         # attached before the offer, as Ring's docs require
        def on_track(track):
            if track.kind == "video":
                self._task = asyncio.ensure_future(self._consume(track))

        @self.pc.on("connectionstatechange")
        async def on_state():
            log.info("ring %s webrtc %s", self.device_id, self.pc.connectionState)
            if self.pc.connectionState in ("failed", "closed"):
                await self.close()

        self.pc.addTransceiver("video", direction="recvonly")
        await self.pc.setLocalDescription(await self.pc.createOffer())   # aiortc gathers ICE before returning
        url = f"{self.client.api}/v1/devices/{self.device_id}/media/streaming/whep/sessions"
        async with httpx.AsyncClient(timeout=20) as http:
            r = await http.post(url, content=self.pc.localDescription.sdp,
                                headers=self.client._headers(**{"Content-Type": "application/sdp"}))
        if r.status_code != 201:
            await self.pc.close()
            raise RuntimeError(f"WHEP {r.status_code}: {r.text[:300]}")
        self.session_url = r.headers.get("Location")
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=r.text, type="answer"))
        self.started = time.monotonic()

    async def _consume(self, track):
        from aiortc.mediastreams import MediaStreamError
        try:
            while not self._closed and time.monotonic() - self.started < self.max_seconds:
                try:
                    frame = await track.recv()
                except MediaStreamError:
                    break
                self.frames += 1
                res = self.on_frame(frame.to_ndarray(format="bgr24"))
                if asyncio.iscoroutine(res):
                    await res
        finally:
            await self.close()

    async def close(self):
        if self._closed:
            return
        self._closed = True
        if self.session_url:
            try:
                async with httpx.AsyncClient(timeout=10) as http:
                    await http.delete(self.session_url, headers=self.client._headers())
            except httpx.HTTPError as e:
                log.warning("closing WHEP session failed: %s", e)
        if self.pc:
            await self.pc.close()
        if self.on_end:
            res = self.on_end(self)
            if asyncio.iscoroutine(res):
                await res
