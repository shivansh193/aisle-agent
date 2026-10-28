import hashlib
import hmac
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aisle_agent import ring
from aisle_agent.models import StubEngine
from aisle_agent.server import create_app

CAT = [{"sku": "a", "name": "packet of biscuits", "price": 10}]
KEY = "test-signing-key"
MOTION = {"meta": {"version": "1.1", "time": "2026-10-20T12:00:00Z", "request_id": "req-1", "account_id": "acc"},
          "data": {"id": "evt-1", "type": "motion_detected", "subType": "human",
                   "attributes": {"source": "dev-1", "source_type": "devices", "timestamp": 1699457230000}}}


def sign(raw: bytes, key=KEY):
    return "sha256=" + hmac.new(key.encode(), raw, hashlib.sha256).hexdigest()


def test_verify_signature_matches_ring_spec():
    raw = json.dumps(MOTION).encode()
    assert ring.verify_signature(KEY, raw, sign(raw))
    assert not ring.verify_signature(KEY, raw + b" ", sign(raw))       # raw bytes, not re-serialised JSON
    assert not ring.verify_signature(KEY, raw, sign(raw, "other"))
    assert not ring.verify_signature(KEY, raw, None)


def test_parse_event():
    ev = ring.parse_event(MOTION)
    assert ev == {"request_id": "req-1", "account_id": "acc", "event_id": "evt-1", "type": "motion_detected",
                  "sub_type": "human", "device_id": "dev-1", "timestamp": 1699457230000}


def test_devices_parses_json_api(monkeypatch):
    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"type": "devices", "id": "dev-1", "attributes": {"name": "Shop shelf", "online": True}}]}

    seen = {}

    def fake_get(url, headers, timeout):
        seen.update(url=url, auth=headers["Authorization"])
        return R()

    monkeypatch.setattr(ring.httpx, "get", fake_get)
    out = ring.RingClient("tok").devices()
    assert out[0]["id"] == "dev-1" and out[0]["name"] == "Shop shelf"
    assert seen == {"url": "https://api.amazonvision.com/v1/devices", "auth": "Bearer tok"}


def test_client_needs_credentials(monkeypatch):
    for k in ("RING_ACCESS_TOKEN", "RING_REFRESH_TOKEN", "RING_CLIENT_ID", "RING_CLIENT_SECRET"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError):
        ring.RingClient.from_env()


class FakeStream:
    def __init__(self):
        self.session_url, self._closed, self.frames = "https://api.amazonvision.com/v1/devices/dev-1/media/x", False, 0

    async def close(self):
        self._closed = True


def test_webhook_signature_dedupe_and_motion_starts_stream(monkeypatch):
    monkeypatch.delenv("AISLE_KEY", raising=False)
    monkeypatch.setenv("RING_HMAC_KEY", KEY)
    opened = []

    async def fake_open(self, device_id, on_frame, on_end=None, max_seconds=60.0):
        opened.append(device_id)
        on_frame(np.zeros((360, 640, 3), np.uint8))
        return FakeStream()

    monkeypatch.setattr(ring.RingClient, "open_stream", fake_open)
    with TestClient(create_app(StubEngine(lambda t: 0.1))) as c:
        s = c.post("/v1/sessions", json={"catalogue": CAT}).json()
        r = c.post(f"/v1/sessions/{s['session_id']}/ring", json={"device_id": "dev-1", "start": "on_motion"},
                   headers={"X-Ring-Token": "tok"})
        assert r.status_code == 200 and r.json()["streaming"] is False and opened == []
        raw = json.dumps(MOTION).encode()
        assert c.post("/v1/ring/webhook", content=raw, headers={"X-Signature": sign(raw, "wrong")}).status_code == 401
        r = c.post("/v1/ring/webhook", content=raw, headers={"X-Signature": sign(raw)})
        assert r.json() == {"status": "processed", "sessions_started": 1}
        assert c.post("/v1/ring/webhook", content=raw, headers={"X-Signature": sign(raw)}).json()["status"] == "duplicate"
        import time
        time.sleep(0.3)
        assert opened == ["dev-1"]
        st = c.get(f"/v1/sessions/{s['session_id']}").json()
        assert st["counts"]["received"] >= 1                       # Ring frames reach the session
        r = c.post(f"/v1/sessions/{s['session_id']}/ring", json={"device_id": "dev-1"}, headers={"X-Ring-Token": "tok"})
        assert r.json()["streaming"] is True and opened == ["dev-1", "dev-1"]
        assert c.delete(f"/v1/sessions/{s['session_id']}/ring").json() == {"ok": True}


def test_webhook_needs_key_and_bypasses_api_key(monkeypatch):
    monkeypatch.setenv("AISLE_KEY", "k1")
    monkeypatch.delenv("RING_HMAC_KEY", raising=False)
    with TestClient(create_app(StubEngine(lambda t: 0.1))) as c:
        assert c.post("/v1/ring/webhook", content=b"{}").status_code == 503     # not 401: no API key needed
        assert c.get("/v1/ring/devices").status_code == 401                    # other Ring calls need the API key


def test_capped_live_view_is_reopened_but_bounded(monkeypatch):
    monkeypatch.delenv("AISLE_KEY", raising=False)
    opened = []

    class Ending:
        def __init__(self):
            self.session_url, self._closed, self.frames = "u", True, 3

        async def close(self):
            pass

    async def fake_open(self, device_id, on_frame, on_end=None, max_seconds=60.0):
        opened.append(device_id)
        s = Ending()
        if len(opened) < 100:
            import asyncio
            asyncio.get_running_loop().call_soon(lambda: asyncio.ensure_future(on_end(s)))   # Ring ends it at once
        return s

    monkeypatch.setattr(ring.RingClient, "open_stream", fake_open)
    with TestClient(create_app(StubEngine(lambda t: 0.1))) as c:
        s = c.post("/v1/sessions", json={"catalogue": CAT}).json()
        r = c.post(f"/v1/sessions/{s['session_id']}/ring", json={"device_id": "dev-1", "start": "continuous"},
                   headers={"X-Ring-Token": "tok"})
        assert r.status_code == 200
        import time
        time.sleep(0.5)
        assert len(opened) == 31                      # first view + 30 reopens, then it stops
        assert c.post(f"/v1/sessions/{s['session_id']}/ring", json={"device_id": "dev-1", "start": "sometimes"},
                      headers={"X-Ring-Token": "tok"}).status_code == 400
