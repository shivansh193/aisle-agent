import base64
import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

from aisle_agent import ring_link
from aisle_agent.models import StubEngine
from aisle_agent.ring_link import LinkStore, RingLinker, compute_nonce, link_time_ok, mask_email
from aisle_agent.server import create_app

KEY = "hmac-signing-key"
CAT = [{"sku": "a", "name": "packet of biscuits", "price": 10}]


def test_nonce_matches_rings_algorithm():
    t, acct = "1761000000000", "ava1.ring.account.ABC"
    mac = hmac.new(KEY.encode(), f"{t}:{acct}".encode(), hashlib.sha256).digest()
    assert compute_nonce(KEY, t, acct) == base64.urlsafe_b64encode(mac).rstrip(b"=").decode()
    assert "=" not in compute_nonce(KEY, t, acct)


def test_mask_and_time_window():
    assert mask_email("shivansh@example.com") == "s***h@example.com"
    assert mask_email("ab@x.in") == "a***@x.in"
    now = int(time.time() * 1000)
    assert link_time_ok(str(now - 5000), now) and not link_time_ok(str(now - 601000), now)
    assert not link_time_ok(str(now + 5000), now) and not link_time_ok("soon", now)


class FakeRing:
    """Stands in for oauth.ring.com and api.amazonvision.com."""

    def __init__(self):
        self.calls = []

    class R:
        def __init__(self, body, code=200):
            self.body, self.status_code = body, code

        def json(self):
            return self.body

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw.get("data") or kw.get("json")))
        if url.endswith("/oauth/token"):
            d = kw["data"]
            if d["grant_type"] == "authorization_code":
                return self.R({"access_token": "acc1", "refresh_token": "ref1", "expires_in": 3600})
            return self.R({"access_token": "acc2", "refresh_token": "ref2", "expires_in": 3600})
        if url.endswith("/app-integrations"):
            return self.R({"data": {}})
        raise AssertionError(url)

    def patch(self, url, **kw):
        self.calls.append(("PATCH", url, kw.get("json")))
        return self.R({"data": {"status": "completed"}})

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw.get("headers", {}).get("Authorization")))
        if url.endswith("/v1/users/me"):
            return self.R({"data": {"id": "ava1.ring.account.ABC"}})
        if url.endswith("/v1/devices"):
            return self.R({"data": [{"id": "dev-1", "attributes": {"name": "Shop counter", "online": True}}]})
        raise AssertionError(url)


@pytest.fixture
def ring(monkeypatch):
    fake = FakeRing()
    for mod in (ring_link.httpx, ):
        monkeypatch.setattr(mod, "post", fake.post)
        monkeypatch.setattr(mod, "get", fake.get)
        monkeypatch.setattr(mod, "patch", fake.patch)
    monkeypatch.setenv("RING_CLIENT_ID", "cid")
    monkeypatch.setenv("RING_CLIENT_SECRET", "secret")
    monkeypatch.setenv("RING_HMAC_KEY", KEY)
    return fake


def test_linking_end_to_end_through_the_api(ring, tmp_path, monkeypatch):
    monkeypatch.setenv("AISLE_KEY", "shop-key")
    store = LinkStore(str(tmp_path / "links.json"))
    with TestClient(create_app(StubEngine(lambda t: 0.1), link_store=store)) as c:
        # 1. Ring's server posts the code to the Token Exchange URL (no Aisle Agent key on this call)
        assert c.post("/ring/oauth/token", data={"code": "authcode"}).json() == {"status": "ok"}
        assert "ava1.ring.account.ABC" in store.unclaimed
        # 2. the owner's browser arrives at the Account Link URL
        t = str(int(time.time() * 1000))
        nonce = compute_nonce(KEY, t, "ava1.ring.account.ABC")
        page = c.get(f"/ring/link?nonce={nonce}&time={t}")
        assert page.status_code == 200 and "Connect your Ring cameras" in page.text and '"ready"' in page.text
        assert c.get(f"/ring/link?nonce={nonce}&time=1").status_code == 400
        # 3. wrong key, wrong nonce, then the real thing
        body = {"nonce": nonce, "time": t, "shop_name": "Kalra Stores", "email": "owner@example.com"}
        assert c.post("/ring/link/complete", json=dict(body, key="nope")).status_code == 401
        assert c.post("/ring/link/complete", json=dict(body, key="shop-key", nonce="forged")).status_code == 404
        r = c.post("/ring/link/complete", json=dict(body, key="shop-key")).json()
        assert r["name"] == "Kalra Stores" and r["account_identifier"] == "o***r@example.com"
        posted = [x for x in ring.calls if x[0] == "POST" and x[1].endswith("/app-integrations")][0][2]
        assert posted == {"account_identifier": "o***r@example.com", "nonce": nonce}
        assert ("PATCH", "https://api.amazonvision.com/v1/accounts/me/app-integrations", {"status": "completed"}) in ring.calls
        assert store.unclaimed == {}
        # 4. the linked shop's cameras, through the stored token
        accts = c.get("/v1/ring/accounts", headers={"Authorization": "Bearer shop-key"}).json()
        shop = accts["shops"][0]["shop_id"]
        assert "tokens" not in json.dumps(accts)                                  # never sent to the browser
    # the link survives a restart
    again = LinkStore(str(tmp_path / "links.json"))
    assert again.shops[shop]["name"] == "Kalra Stores" and again.shops[shop]["tokens"]["refresh_token"] == "ref1"


def test_linked_client_refreshes_and_saves_rotated_tokens(ring, tmp_path, monkeypatch):
    from aisle_agent import ring as ring_mod
    monkeypatch.setattr(ring_mod.httpx, "post", ring.post)
    monkeypatch.setattr(ring_mod.httpx, "get", ring.get)
    store = LinkStore(str(tmp_path / "links.json"))
    store.shops["shop_1"] = {"name": "S", "account_identifier": "o***r@example.com", "account_id": "acct", "linked_at": 0,
                             "tokens": {"access_token": "old", "refresh_token": "ref1", "expires_at": 0}}
    linker = RingLinker(store)
    devices = linker.client("shop_1").devices()                                  # expired: refreshes first
    assert devices[0]["name"] == "Shop counter"
    assert store.shops["shop_1"]["tokens"]["access_token"] == "acc2"
    assert LinkStore(str(tmp_path / "links.json")).shops["shop_1"]["tokens"]["refresh_token"] == "ref2"
    assert ("GET", "https://api.amazonvision.com/v1/devices", "Bearer acc2") in ring.calls


def test_app_integration_removed_unlinks(ring, tmp_path, monkeypatch):
    monkeypatch.delenv("AISLE_KEY", raising=False)
    store = LinkStore(str(tmp_path / "links.json"))
    store.shops["shop_1"] = {"name": "S", "account_identifier": "x", "account_id": "acct-9", "linked_at": 0,
                             "tokens": {"access_token": "a", "refresh_token": "r", "expires_at": 0}}
    payload = {"meta": {"version": "1.1", "time": "t", "request_id": "rq-77", "account_id": "acct-9"},
               "data": {"id": "e", "type": "app_integration_removed", "attributes": {}}}
    raw = json.dumps(payload).encode()
    sig = "sha256=" + hmac.new(KEY.encode(), raw, hashlib.sha256).hexdigest()
    with TestClient(create_app(StubEngine(lambda t: 0.1), link_store=store)) as c:
        assert c.post("/v1/ring/webhook", content=raw, headers={"X-Signature": sig}).json() == {"status": "processed", "unlinked": True}
    assert store.shops == {}


def test_token_exchange_needs_credentials(monkeypatch, tmp_path):
    for k in ("RING_CLIENT_ID", "RING_CLIENT_SECRET", "RING_HMAC_KEY"):
        monkeypatch.delenv(k, raising=False)
    with TestClient(create_app(StubEngine(lambda t: 0.1), link_store=LinkStore(str(tmp_path / "l.json")))) as c:
        assert c.post("/ring/oauth/token", data={"code": "x"}).status_code == 503
