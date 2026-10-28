"""WHIP (WebRTC-HTTP ingestion) on aiortc: the client POSTs an SDP offer, gets an SDP answer,
and its video track is fed frame by frame into the session (newest frame wins).

No trickle ICE: the client sends its offer after ICE gathering, and aiortc gathers fully
before returning the answer. Behind NAT, or on hosts without inbound UDP, set a TURN server
through AISLE_ICE (JSON list of RTCIceServer dicts).
"""
import asyncio
import json
import logging
import os

log = logging.getLogger("aisle_agent.webrtc")


def ice_servers():
    from aiortc import RTCIceServer
    raw = os.environ.get("AISLE_ICE")
    servers = json.loads(raw) if raw else [{"urls": "stun:stun.l.google.com:19302"}]
    return [RTCIceServer(**s) for s in servers]


class WhipPeer:
    def __init__(self, session):
        from aiortc import RTCConfiguration, RTCPeerConnection
        self.session = session
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers()))
        self.tasks = []

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                self.tasks.append(asyncio.ensure_future(self._consume(track)))

        @self.pc.on("connectionstatechange")
        async def on_state():
            log.info("session %s webrtc %s", session.id, self.pc.connectionState)
            session.publish([{"type": "transport", "state": self.pc.connectionState}])
            if self.pc.connectionState in ("failed", "closed"):
                await self.close()

    async def answer(self, offer_sdp: str) -> str:
        from aiortc import RTCSessionDescription
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
        await self.pc.setLocalDescription(await self.pc.createAnswer())
        return self.pc.localDescription.sdp

    async def _consume(self, track):
        from aiortc.mediastreams import MediaStreamError
        gap, last = 0.9 / self.session.opt.max_hz, -1.0
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                return
            now = self.session.now()
            if now - last < gap:                 # no point converting frames the worker would drop
                continue
            last = now
            self.session.offer(frame.to_ndarray(format="bgr24"), now)

    async def close(self):
        for t in self.tasks:
            t.cancel()
        await self.pc.close()
