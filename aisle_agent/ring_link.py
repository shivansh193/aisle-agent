"""Linking a shop owner's own Ring account: Ring's one-way (Ring-driven) account linking.

The owner adds Aisle Agent in the Ring app (Ring App Store, or a private app's invite for a
pilot of up to 5 accounts). Then:

1. Token Exchange URL (server to server). Ring POSTs `code` (form-encoded). We exchange it at
   https://oauth.ring.com/oauth/token within 60 s, read the Ring account id from
   GET /v1/users/me, and hold the tokens as UNCLAIMED under that account id.
2. Account Link URL (the owner's browser). Ring redirects with `?nonce=&time=` (time in ms,
   valid 600 s). The owner proves this is their shop (here: the deployment's Aisle Agent key).
3. Complete. Find the unclaimed token whose account id reproduces the nonce:
       nonce = base64url_nopad(HMAC-SHA256(hmac_signing_key, f"{time}:{account_id}"))
   then confirm with Ring: POST /v1/accounts/me/app-integrations {account_identifier, nonce},
   PATCH {status: "completed"}. The token is now CLAIMED by the shop.
4. From then on, the refresh token keeps access alive (Ring may rotate it; we save the new one),
   GET /v1/devices lists the shared cameras, live view is WHEP (ring.py), and events arrive on
   the webhook with meta.account_id. `app_integration_removed` drops the link.

Tokens are kept in a JSON file (RING_LINK_STORE, default ring_links.json) readable only by the
server user. Credentials come from the Ring Developer Portal: RING_CLIENT_ID,
RING_CLIENT_SECRET, RING_HMAC_KEY.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from typing import Dict, Optional

import httpx

from .ring import API, OAUTH, RingClient

LINK_WINDOW_S = 600


def compute_nonce(signing_key: str, time_param: str, account_id: str) -> str:
    mac = hmac.new(signing_key.encode(), f"{time_param}:{account_id}".encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).rstrip(b"=").decode()


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    if not domain:
        raise ValueError("not an email address")
    masked = local[0] + "***" if len(local) <= 2 else local[0] + "***" + local[-1]
    return f"{masked}@{domain}"


def link_time_ok(time_param: str, now_ms: Optional[int] = None) -> bool:
    try:
        t = int(time_param)
    except (TypeError, ValueError):
        return False
    delta = ((now_ms if now_ms is not None else int(time.time() * 1000)) - t) / 1000
    return 0 <= delta <= LINK_WINDOW_S


class LinkStore:
    """Unclaimed tokens (by Ring account id) and linked shops (by shop id), saved to a JSON file."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.environ.get("RING_LINK_STORE", "ring_links.json")
        self.lock = threading.Lock()
        self.unclaimed: Dict[str, dict] = {}
        self.shops: Dict[str, dict] = {}
        if os.path.exists(self.path):
            data = json.load(open(self.path, encoding="utf-8"))
            self.unclaimed, self.shops = data.get("unclaimed", {}), data.get("shops", {})

    def save(self):
        tmp = self.path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"unclaimed": self.unclaimed, "shops": self.shops}, f)
        os.replace(tmp, self.path)

    def prune(self, max_age_s: float = 3600):
        cutoff = time.time() - max_age_s
        self.unclaimed = {k: v for k, v in self.unclaimed.items() if v["created_at"] > cutoff}

    def public(self, shop_id: str) -> dict:
        s = self.shops[shop_id]
        return {"shop_id": shop_id, "name": s["name"], "account_identifier": s["account_identifier"],
                "linked_at": s["linked_at"], "ring_account": s["account_id"][:10] + "…"}

    def by_account(self, account_id: str) -> Optional[str]:
        return next((sid for sid, s in self.shops.items() if s["account_id"] == account_id), None)


class RingLinker:
    def __init__(self, store: LinkStore, client_id: Optional[str] = None, client_secret: Optional[str] = None,
                 signing_key: Optional[str] = None, api: str = API, oauth: str = OAUTH):
        self.store, self.api, self.oauth = store, api, oauth
        self.client_id = client_id or os.environ.get("RING_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("RING_CLIENT_SECRET")
        self.signing_key = signing_key or os.environ.get("RING_HMAC_KEY")

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret and self.signing_key)

    # -- 1. Token Exchange URL
    def exchange_code(self, code: str) -> str:
        r = httpx.post(self.oauth, data={"grant_type": "authorization_code", "code": code,
                                         "client_id": self.client_id, "client_secret": self.client_secret}, timeout=15)
        r.raise_for_status()
        tok = r.json()
        me = httpx.get(f"{self.api}/v1/users/me", headers={"Authorization": f"Bearer {tok['access_token']}"}, timeout=15)
        me.raise_for_status()
        account_id = me.json()["data"]["id"]
        with self.store.lock:
            self.store.prune()
            self.store.unclaimed[account_id] = {
                "access_token": tok["access_token"], "refresh_token": tok.get("refresh_token"),
                "expires_at": time.time() + tok.get("expires_in", 3600) - 60, "account_id": account_id,
                "created_at": time.time()}
            self.store.save()
        return account_id

    # -- 3. Complete linking
    def complete(self, nonce: str, time_param: str, shop_name: str, email: str) -> dict:
        if not link_time_ok(time_param):
            raise ValueError("this link request has expired; start again from the Ring app")
        with self.store.lock:
            rec = next((r for aid, r in self.store.unclaimed.items()
                        if hmac.compare_digest(compute_nonce(self.signing_key, time_param, aid), nonce)), None)
        if rec is None:
            raise LookupError("no pending Ring authorization matches this link; try again from the Ring app")
        auth = {"Authorization": f"Bearer {rec['access_token']}", "Content-Type": "application/json"}
        r = httpx.post(f"{self.api}/v1/accounts/me/app-integrations", headers=auth,
                       json={"account_identifier": mask_email(email), "nonce": nonce}, timeout=15)
        r.raise_for_status()
        r = httpx.patch(f"{self.api}/v1/accounts/me/app-integrations", headers=auth, json={"status": "completed"}, timeout=15)
        r.raise_for_status()
        with self.store.lock:
            shop_id = self.store.by_account(rec["account_id"]) or "shop_" + secrets.token_hex(4)
            self.store.shops[shop_id] = {"name": shop_name.strip()[:80] or "My shop", "account_identifier": mask_email(email),
                                         "account_id": rec["account_id"], "linked_at": int(time.time()),
                                         "tokens": {k: rec[k] for k in ("access_token", "refresh_token", "expires_at")}}
            self.store.unclaimed.pop(rec["account_id"], None)
            self.store.save()
        return self.store.public(shop_id)

    # -- after linking
    def client(self, shop_id: str) -> RingClient:
        if shop_id not in self.store.shops:
            raise KeyError("no such linked shop")
        tok = self.store.shops[shop_id]["tokens"]

        def saved(access, refresh, expires_at):        # keep rotated tokens across restarts
            with self.store.lock:
                if shop_id in self.store.shops:
                    self.store.shops[shop_id]["tokens"] = {"access_token": access, "refresh_token": refresh,
                                                           "expires_at": expires_at}
                    self.store.save()

        c = RingClient(tok["access_token"], tok.get("refresh_token"), self.client_id, self.client_secret,
                       api=self.api, on_refresh=saved)
        c._expires = tok.get("expires_at", 0)
        return c

    def unlink_account(self, account_id: str) -> Optional[str]:
        with self.store.lock:
            sid = self.store.by_account(account_id)
            if sid:
                del self.store.shops[sid]
                self.store.save()
        return sid
