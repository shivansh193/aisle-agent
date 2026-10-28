"""Aisle Agent live API.

POST   /v1/sessions                       {catalogue: [{sku, name, price}], zones?: [{id, box, skus}],
                                           currency?, options?} -> session
PUT    /v1/sessions/{id}/zones            [{id, box: [l, t, r, b] fractions, skus}] (empty list = hold mode)
GET    /v1/ice                            STUN/TURN servers for the browser
GET    /v1/sessions/{id}                  stats (rates, stage timings, counts)
DELETE /v1/sessions/{id}
POST   /v1/sessions/{id}/whip             WebRTC ingest (body: SDP offer, application/sdp) -> 201 SDP answer
WS     /v1/sessions/{id}/ingest           fallback ingest: binary JPEG frames over a WebSocket
POST   /v1/sessions/{id}/frames?t=        one JPEG frame, processed now; returns its events (replay / tests)
WS     /v1/sessions/{id}/events           take / line / state / error events as JSON
GET    /v1/sessions/{id}/bill
POST   /v1/sessions/{id}/bill/{line}/dispute   flags a line for the exit check (never removes it)
GET    /v1/sessions                       live sessions with their totals and lines to check
GET    /v1/sessions/{id}/bill/{line}/evidence   before/after hand crops, scores, how it was named
POST   /v1/sessions/{id}/bill/{line}/review     {action: approve | remove, by?, note?}: the exit check
GET    /exit                              staff exit-check page
GET    /cameras                           linked Ring accounts and cameras
GET    /v1/sessions/{id}/preview.mjpg     live feed: the frames the pipeline analyses, as MJPEG (any source)
GET    /v1/sessions/{id}/preview.jpg      the latest one
GET    /v1/sessions/{id}/shelves          shelf watch: stock level per zone, history, open restock alerts
GET    /                                   demo page (phone or laptop camera -> live bill)

Shopper's bill (no shop key; an unguessable link per session, shown as a QR code on the console):
GET    /b/{token}                         the shopper's phone page
GET    /p/{token}/bill   WS /p/{token}/events   their bill, live
POST   /p/{token}/lines/{line}/not-mine   dispute a line (it goes to the exit check)

Shop layout (layout.py; saved to AISLE_LAYOUT):
GET    /layout                            floor-plan editor: shelves, products, stock
GET    /v1/layout   PUT /v1/layout        the layout with each shelf's state (ok | low | empty | alert)
POST   /v1/layout/shelves/{id}/restock    {sku, qty} set a count after refilling (or {} to clear an alert)
GET    /v1/layout/catalogue               products, and the products on each zone, for the console

Ring cameras (see ring.py):
GET    /v1/ring/devices                   the linked Ring cameras
POST   /v1/sessions/{id}/ring             {device_id, start: "now" | "on_motion"} pull the camera's live
                                           view (WHEP) into this session, now or on its next motion event
DELETE /v1/sessions/{id}/ring
POST   /v1/ring/webhook                   Ring event webhook; X-Signature verified with RING_HMAC_KEY
Ring auth: RING_ACCESS_TOKEN (or refresh token + client credentials) in the environment, or an
`X-Ring-Token` header (e.g. a Playground token) on the calls above.

If AISLE_KEY is set, every call needs `Authorization: Bearer <key>` or `?key=<key>`.
"""
import asyncio
import json
import os
import secrets
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

from .logic import Catalogue
from .pipeline import Options, Session

STATIC = Path(__file__).parent / "static"


def decode_jpeg(data: bytes) -> np.ndarray:
    import cv2
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a decodable image")
    return img



def _plain(o):
    if isinstance(o, np.generic):          # model outputs reach events as numpy scalars
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def dumps(msg) -> str:
    return json.dumps(msg, default=_plain)

class Agents:
    """The Token Factory client shared by the exit advisor and the shop assistant."""

    def __init__(self, client, vl_backend=None):
        from .agents import ExitAdvisor
        self.client = client
        self.advisor = ExitAdvisor(client, vl_backend)
        self.assistant = None                          # built inside create_app, where the tools' data lives

    def trace(self) -> dict:
        return {"engine": "agents (Nemotron-3-Super, Nemotron-3.5-Lightning)", "usage": dict(self.client.usage),
                "calls": list(self.client.trace)[-50:]}


def make_agents(engine) -> Optional[Agents]:
    """Agents when a Token Factory key is available (shares the Nemotron engine's client when there is one)."""
    try:
        from vjev.tokenfactory import TokenFactoryBackend, TokenFactoryClient, api_key
    except ImportError:
        return None
    vl = getattr(engine, "backend", None)
    if vl is not None and hasattr(vl, "client"):
        return Agents(vl.client, vl)
    if not api_key():
        return None
    client = TokenFactoryClient()
    return Agents(client, TokenFactoryBackend(client=client))


def create_app(engine, max_sessions: int = 4, link_store=None, layout=None, agents=None) -> FastAPI:
    app = FastAPI(title="Aisle Agent live", version="0.1.0")
    sessions: dict = {}
    peers: dict = {}
    shares: dict = {}                      # shopper link token -> session id
    from .layout import Layout
    shop = layout or Layout()
    from collections import deque
    events_log: deque = deque(maxlen=500)          # recent takes, put-backs, unchecked visits, alerts (all sessions)
    decisions: List[dict] = []                     # exit-check decisions with the agent's advice
    app.state.agents = agents if agents is not None else make_agents(engine)

    def log_events(sid, msgs):
        for m in msgs:
            if m.get("type") in ("take", "put", "line", "unchecked", "restock", "dispute", "review"):
                e = {k: v for k, v in m.items() if k not in ("bill", "candidates")}
                if "line" in e:
                    e["line"] = {k: e["line"].get(k) for k in ("id", "name", "sku", "status", "zone", "confidence", "note")}
                events_log.append({"session": sid, "at": round(time.time(), 1), **e})
    key = os.environ.get("AISLE_KEY")

    def authorised(headers, query) -> bool:
        if not key:
            return True
        auth = headers.get("authorization", "")
        given = auth[7:] if auth.lower().startswith("bearer ") else query.get("key", "")
        return secrets.compare_digest(given, key)

    @app.middleware("http")
    async def auth_mw(request: Request, call_next):
        if (request.url.path.startswith("/v1/") and request.url.path != "/v1/ring/webhook"   # webhook: HMAC instead
                and not authorised(request.headers, request.query_params)):
            return Response(status_code=401, content="missing or wrong key")
        return await call_next(request)

    def get(sid: str) -> Session:
        if sid not in sessions:
            raise HTTPException(404, "no such session")
        return sessions[sid]

    @app.get("/health")
    def health():
        return {"ok": True, "sessions": len(sessions), "engine": type(engine).__name__,
                "engine_label": getattr(engine, "label", None)}

    @app.get("/v1/models/trace")
    def model_trace():
        """Every hosted-model call: model, purpose, latency, tokens, cost (for the console's model trace)."""
        calls = []
        for src in (engine, getattr(app.state, "agents", None)):
            if src is not None and hasattr(src, "trace"):
                calls.append(src.trace())
        return {"engine": getattr(engine, "label", type(engine).__name__), "sources": calls}

    @app.get("/", response_class=HTMLResponse)
    def demo_page():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/exit", response_class=HTMLResponse)
    def exit_page():
        return (STATIC / "exit.html").read_text(encoding="utf-8")

    @app.get("/layout", response_class=HTMLResponse)
    def layout_page():
        return (STATIC / "layout.html").read_text(encoding="utf-8")

    @app.get("/v1/layout")
    def layout_get():
        return shop.view()

    @app.put("/v1/layout")
    def layout_put(body: dict = Body(...)):
        try:
            return shop.replace(body)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.post("/v1/layout/shelves/{shelf_id}/restock")
    def layout_restock(shelf_id: str, body: dict = Body(default={})):
        try:
            return shop.restock(shelf_id, body.get("sku"), None if body.get("qty") is None else int(body["qty"]))
        except KeyError:
            raise HTTPException(404, "no such shelf")
        except (TypeError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/v1/layout/catalogue")
    def layout_catalogue():
        return shop.catalogue()

    @app.get("/cameras", response_class=HTMLResponse)
    def cameras_page():
        return (STATIC / "cameras.html").read_text(encoding="utf-8")

    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/v1/sessions")
    def list_sessions():
        out = []
        for sid, s in sessions.items():
            b = s.bill.as_dict()
            out.append({"session_id": sid, "mode": s.mode, "uptime_s": round(s.now(), 1), "lines": len(b["lines"]),
                        "to_verify": len(b["to_verify"]), "total": b["total"], "currency": b["currency"]})
        return {"sessions": out}

    @app.post("/v1/sessions", status_code=201)
    def create(body: dict, request: Request):
        if len(sessions) >= max_sessions:
            raise HTTPException(429, f"at most {max_sessions} live sessions")
        sid = secrets.token_urlsafe(8)
        try:
            cat = Catalogue(body.get("catalogue") or [])
            opt = Options(**(body.get("options") or {}))
            session = Session(sid, engine, cat, opt, body.get("currency", "INR"), body.get("zones"))
        except (TypeError, ValueError, OSError) as e:
            raise HTTPException(400, str(e))
        sessions[sid] = session
        session.share = secrets.token_urlsafe(12)
        session.listeners.append(shop.observe)
        session.listeners.append(log_events)
        shares[session.share] = sid
        base = f"/v1/sessions/{sid}"
        return {"session_id": sid, "whip_url": f"{base}/whip", "ingest_ws": f"{base}/ingest",
                "events_ws": f"{base}/events", "frames_url": f"{base}/frames", "bill_url": f"{base}/bill",
                "products": len(cat), "mode": session.mode, "shopper_url": f"/b/{session.share}"}

    @app.put("/v1/sessions/{sid}/zones")
    def zones(sid: str, zones: list = Body(...)):
        s = get(sid)
        try:
            s.set_zones(zones)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, str(e))
        return {"mode": s.mode, "zones": len(s.zones)}

    @app.get("/v1/ice")
    def ice():
        """ICE servers for the browser (the same STUN/TURN list the server uses)."""
        import json
        raw = os.environ.get("AISLE_ICE")
        return {"iceServers": json.loads(raw) if raw else [{"urls": "stun:stun.l.google.com:19302"}]}

    # ------------------------------------------------------------------ Ring
    from .ring_link import LinkStore, RingLinker, link_time_ok
    linker = RingLinker(link_store or LinkStore())
    ring_bind: dict = {}           # sid -> {"client", "device_id", "mode", "stream", "reopens"}
    seen_requests: list = []
    KEEP_WATCHING_S = 5.0          # reopen a capped live view while someone was seen this recently
    MAX_REOPENS = 30

    def ring_client(request: Request, shop: Optional[str] = None):
        from .ring import RingClient
        try:
            if shop:
                return linker.client(shop)
            return RingClient.from_env(request.headers.get("x-ring-token"))
        except KeyError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))

    async def start_ring_stream(sid: str):
        b, s = ring_bind[sid], sessions.get(sid)
        if s is None or (b.get("stream") and not b["stream"]._closed):
            return b.get("stream")

        async def ended(stream):
            s.publish([{"type": "ring_stream", "state": "ended", "device_id": b["device_id"], "frames": stream.frames}])
            # Ring caps a live view at 30 s (battery) / 60 s (wired): keep watching while it matters
            people = any(s.now() - ts <= KEEP_WATCHING_S for ts, _ in s.last_seen.values())
            keep = b["mode"] == "continuous" or (b["mode"] == "on_motion" and people)
            if keep and ring_bind.get(sid) is b and sid in sessions and b["reopens"] < MAX_REOPENS:
                b["reopens"] += 1
                try:
                    await start_ring_stream(sid)
                except Exception as e:
                    s.publish([{"type": "error", "message": f"Ring live view: {e}"}])

        b["stream"] = await b["client"].open_stream(b["device_id"], lambda f: s.offer(f), ended)
        s.publish([{"type": "ring_stream", "state": "started", "device_id": b["device_id"], "reopen": b["reopens"]}])
        return b["stream"]

    # -- linking a shop owner's own Ring account (one-way account linking; ring_link.py)
    @app.post("/ring/oauth/token")
    async def ring_token_exchange(request: Request):
        """Token Exchange URL: Ring POSTs the authorization code here (server to server)."""
        if not linker.configured:
            raise HTTPException(503, "Ring client credentials are not configured")
        form = await request.form()
        if not form.get("code"):
            raise HTTPException(400, "missing code")
        try:
            await asyncio.to_thread(linker.exchange_code, form["code"])
        except Exception as e:
            raise HTTPException(502, f"token exchange failed: {e}")
        return {"status": "ok"}

    @app.get("/ring/link", response_class=HTMLResponse)
    def ring_link_page(nonce: str = "", time: str = ""):
        """Account Link URL: Ring sends the owner's browser here with ?nonce=&time=."""
        if not nonce or not link_time_ok(time):
            return HTMLResponse((STATIC / "link.html").read_text(encoding="utf-8").replace("{{state}}", "expired"), 400)
        return (STATIC / "link.html").read_text(encoding="utf-8").replace("{{state}}", "ready")

    @app.post("/ring/link/complete")
    async def ring_link_complete(body: dict):
        if key and not secrets.compare_digest(str(body.get("key", "")), key):
            raise HTTPException(401, "that is not this shop's Aisle Agent key")
        missing = [f for f in ("nonce", "time", "shop_name", "email") if not body.get(f)]
        if missing:
            raise HTTPException(400, f"missing {', '.join(missing)}")
        try:
            return await asyncio.to_thread(linker.complete, body["nonce"], str(body["time"]), body["shop_name"], body["email"])
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(502, f"Ring did not confirm the link: {e}")

    @app.get("/v1/ring/accounts")
    def ring_accounts():
        return {"configured": linker.configured, "shops": [linker.store.public(sid) for sid in linker.store.shops]}

    @app.delete("/v1/ring/accounts/{shop_id}")
    def ring_forget(shop_id: str):
        """Forget a linked shop here. (The owner removes Aisle Agent in the Ring app to revoke it at Ring.)"""
        with linker.store.lock:
            if linker.store.shops.pop(shop_id, None) is None:
                raise HTTPException(404, "no such linked shop")
            linker.store.save()
        return {"ok": True}

    @app.get("/v1/ring/devices")
    async def ring_devices(request: Request, shop: Optional[str] = None):
        client = ring_client(request, shop)
        try:
            return {"devices": [{k: d[k] for k in ("id", "name", "online")} for d in await asyncio.to_thread(client.devices)]}
        except Exception as e:                       # httpx errors carry Ring's status and message
            raise HTTPException(502, f"Ring: {e}")

    @app.post("/v1/sessions/{sid}/ring")
    async def ring_attach(sid: str, body: dict, request: Request):
        get(sid)
        if not body.get("device_id"):
            raise HTTPException(400, "device_id is required")
        mode = body.get("start", "now")
        if mode not in ("now", "on_motion", "continuous"):
            raise HTTPException(400, "start must be 'now', 'on_motion' or 'continuous'")
        old = ring_bind.get(sid, {}).get("stream")
        if old:
            await old.close()
        ring_bind[sid] = {"client": ring_client(request, body.get("shop")), "device_id": body["device_id"], "mode": mode,
                          "stream": None, "reopens": 0, "shop": body.get("shop")}
        if mode in ("now", "continuous"):
            try:
                await start_ring_stream(sid)
            except Exception as e:
                raise HTTPException(502, f"Ring live view: {e}")
        return {"device_id": body["device_id"], "start": mode,
                "streaming": bool(ring_bind[sid]["stream"]), "session_url": getattr(ring_bind[sid]["stream"], "session_url", None)}

    @app.delete("/v1/sessions/{sid}/ring")
    async def ring_detach(sid: str):
        b = ring_bind.pop(sid, None)
        if b and b.get("stream"):
            await b["stream"].close()
        return {"ok": True}

    @app.post("/v1/ring/webhook")
    async def ring_webhook(request: Request):
        from .ring import parse_event, verify_signature
        hkey = os.environ.get("RING_HMAC_KEY")
        if not hkey:
            raise HTTPException(503, "RING_HMAC_KEY is not configured")
        raw = await request.body()
        if not verify_signature(hkey, raw, request.headers.get("x-signature")):
            raise HTTPException(401, "bad signature")
        import json
        ev = parse_event(json.loads(raw))
        if ev["request_id"] in seen_requests:
            return {"status": "duplicate"}
        seen_requests.append(ev["request_id"])
        del seen_requests[:-1000]
        if ev["type"] == "app_integration_removed" and ev["account_id"]:
            gone = linker.unlink_account(ev["account_id"])
            for sid, b in list(ring_bind.items()):
                if gone and b.get("shop") == gone:
                    if b.get("stream"):
                        await b["stream"].close()
                    ring_bind.pop(sid, None)
            return {"status": "processed", "unlinked": bool(gone)}
        started = []
        for sid, b in list(ring_bind.items()):
            if b["device_id"] != ev["device_id"] or sid not in sessions:
                continue
            sessions[sid].publish([{"type": "ring_event", **ev}])
            if ev["type"] == "motion_detected" and b["mode"] == "on_motion":
                b["reopens"] = 0
                task = asyncio.get_running_loop().create_task(start_ring_stream(sid))   # reply within Ring's 5 s
                task.add_done_callback(lambda t, sid=sid: t.cancelled() or not t.exception() or sessions.get(sid) and
                                       sessions[sid].publish([{"type": "error", "message": f"Ring live view: {t.exception()}"}]))
                started.append(sid)
        return {"status": "processed", "sessions_started": len(started)}

    @app.get("/v1/sessions/{sid}")
    def stats(sid: str):
        return get(sid).stats()

    @app.delete("/v1/sessions/{sid}", status_code=204)
    async def delete(sid: str):
        s = get(sid)
        b = ring_bind.pop(sid, None)
        if b and b.get("stream"):
            await b["stream"].close()
        if sid in peers:
            await peers.pop(sid).close()
        await s.close()
        shares.pop(getattr(s, "share", None), None)
        sessions.pop(sid, None)
        return Response(status_code=204)

    @app.post("/v1/sessions/{sid}/whip")
    async def whip(sid: str, request: Request):
        s = get(sid)
        if "sdp" not in request.headers.get("content-type", ""):
            raise HTTPException(415, "send the SDP offer as application/sdp")
        from .webrtc import WhipPeer
        if sid in peers:
            await peers.pop(sid).close()
        peer = WhipPeer(s)
        answer = await peer.answer((await request.body()).decode())
        peers[sid] = peer
        return Response(answer, status_code=201, media_type="application/sdp",
                        headers={"Location": f"/v1/sessions/{sid}/whip"})

    @app.delete("/v1/sessions/{sid}/whip", status_code=200)
    async def whip_stop(sid: str):
        get(sid)
        if sid in peers:
            await peers.pop(sid).close()
        return {"ok": True}

    @app.post("/v1/sessions/{sid}/frames")
    async def frame(sid: str, request: Request, t: Optional[float] = None):
        s = get(sid)
        try:
            img = decode_jpeg(await request.body())
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"events": await s.handle(img, t)}

    @app.get("/v1/sessions/{sid}/share")
    def share(sid: str):
        return {"shopper_url": f"/b/{get(sid).share}"}

    # ---- the shopper's own bill: an unguessable link per session, no shop key; read, plus "not mine"

    def shopper_session(token: str) -> Session:
        sid = shares.get(token)
        if sid is None or sid not in sessions:
            raise HTTPException(404, "this bill has closed")
        return sessions[sid]

    def shopper_bill(s: Session) -> dict:
        b = s.bill.as_dict()
        keep = ("id", "name", "price", "status", "note", "seen", "zone", "t")
        return {"currency": b["currency"], "total": b["total"], "to_verify": len(b["to_verify"]),
                "code": s.share[:4].upper(), "lines": [{k: l.get(k) for k in keep} for l in b["lines"]]}

    @app.get("/b/{token}", response_class=HTMLResponse)
    def shopper_page(token: str):
        shopper_session(token)
        return (STATIC / "phone.html").read_text(encoding="utf-8")

    @app.get("/p/{token}/bill")
    def shopper_get(token: str):
        return shopper_bill(shopper_session(token))

    @app.post("/p/{token}/lines/{line_id}/not-mine")
    def shopper_dispute(token: str, line_id: int):
        s = shopper_session(token)
        line = s.bill.lines.get(line_id)
        if line is None:
            raise HTTPException(404, "no such line")
        if line.status not in ("ok", "verify", "approved"):
            raise HTTPException(409, f"this line is already {line.status}")
        s.bill.dispute(line_id)
        s.publish([{"type": "dispute", "by": "shopper", "line": vars(line).copy(), "bill": s.bill.as_dict()}])
        return shopper_bill(s)

    @app.websocket("/p/{token}/events")
    async def shopper_events(ws: WebSocket, token: str):
        sid = shares.get(token)
        if sid not in sessions:
            await ws.close(code=4404)
            return
        s = sessions[sid]
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue()
        s.subscribers.add(q)
        try:
            await ws.send_text(dumps(shopper_bill(s)))
            while True:
                if "bill" in await q.get():            # only bill changes reach the shopper
                    await ws.send_text(dumps(shopper_bill(s)))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            s.subscribers.discard(q)

    @app.get("/v1/sessions/{sid}/bill")
    def bill(sid: str):
        return get(sid).bill.as_dict()

    @app.post("/v1/sessions/{sid}/bill/{line_id}/dispute")
    def dispute(sid: str, line_id: int):
        s = get(sid)
        if line_id not in s.bill.lines:
            raise HTTPException(404, "no such line")
        line = s.bill.dispute(line_id)
        s.publish([{"type": "dispute", "line": vars(line).copy(), "bill": s.bill.as_dict()}])
        return s.bill.as_dict()

    @app.get("/v1/sessions/{sid}/preview.jpg")
    def preview_jpg(sid: str):
        s = get(sid)
        if s.preview is None:
            raise HTTPException(404, "no frame yet")
        return Response(s.preview[1], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/v1/sessions/{sid}/preview.mjpg")
    async def preview_mjpg(sid: str, fps: float = 8.0, frames: int = 0):
        get(sid)

        async def gen():
            # every part ends with the next boundary: browsers paint a part only once they see it
            yield b"--frame\r\n"
            last, sent, at = None, 0, 0.0
            while sid in sessions and (frames <= 0 or sent < frames):
                p, now = sessions[sid].preview, time.monotonic()
                if p is not None and (p[0] != last or now - at > 2.0):     # resend when idle (keepalive)
                    last, sent, at = p[0], sent + 1, now
                    head = f"Content-Type: image/jpeg\r\nContent-Length: {len(p[1])}\r\n\r\n".encode()
                    yield head + p[1] + b"\r\n--frame\r\n"
                await asyncio.sleep(1.0 / max(fps, 0.5))

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/v1/sessions/{sid}/shelves")
    def shelves(sid: str):
        s = get(sid)
        st = s.stock.as_dict()
        return {"zones": {z.id: st.get(z.id) for z in s.zones}, "alerts": [z for z, v in st.items() if v.get("alert")]}

    @app.get("/v1/sessions/{sid}/bill/{line_id}/evidence")
    def evidence(sid: str, line_id: int):
        s = get(sid)
        if line_id not in s.bill.lines:
            raise HTTPException(404, "no such line")
        return {"line": vars(s.bill.lines[line_id]).copy(), **s.evidence.get(line_id, {})}

    def zone_products(s, zone):
        z = next((z for z in s.zones if z.id == zone), None)
        skus = z.skus if z and z.skus else [p.sku for p in s.catalogue.products]
        return [s.catalogue.by_sku[k].name for k in skus if k in s.catalogue.by_sku]

    @app.get("/v1/sessions/{sid}/bill/{line_id}/advice")
    def advice(sid: str, line_id: int):
        """Nemotron-3-Super's explanation and recommendation for a flagged line. Advice only."""
        s = get(sid)
        if line_id not in s.bill.lines:
            raise HTTPException(404, "no such line")
        ag = app.state.agents
        if ag is None:
            raise HTTPException(503, "the exit advisor needs a Token Factory key (TOKEN_FACTORY_API_KEY)")
        line = vars(s.bill.lines[line_id]).copy()
        try:
            return ag.advisor.advise((sid, line_id), line, s.evidence.get(line_id, {}), zone_products(s, line.get("zone")))
        except Exception as e:                         # the advisor being down never blocks the exit check
            raise HTTPException(503, f"advisor unavailable: {str(e)[:120]}")

    @app.get("/v1/agents/stats")
    def agent_stats():
        judged = [d for d in decisions if d.get("advice") in ("keep", "remove")]
        agree = sum(1 for d in judged if d["agrees"])
        return {"decisions": len(decisions), "with_advice": len(judged), "agreed": agree,
                "agreement_rate": round(agree / len(judged), 3) if judged else None, "recent": decisions[-20:]}

    @app.post("/v1/assistant")
    def assistant(body: dict = Body(...)):
        ag = app.state.agents
        if ag is None:
            raise HTTPException(503, "the assistant needs a Token Factory key (TOKEN_FACTORY_API_KEY)")
        q = str(body.get("question", "")).strip()[:600]
        if not q:
            raise HTTPException(400, "ask a question")
        try:
            return ag.assistant.ask(q, body.get("history") or [])
        except Exception as e:
            raise HTTPException(503, f"assistant unavailable: {str(e)[:160]}")

    # ---- read-only tools for the assistant
    def busiest():
        return max(sessions.values(), key=lambda x: len(x.bill.lines), default=None)

    def t_sessions():
        return [{"session": sid, "lines": len(x.bill.lines), "to_verify": len(x.bill.as_dict()["to_verify"]),
                 "total": x.bill.as_dict()["total"]} for sid, x in sessions.items()]

    def t_bill(session: str = ""):
        x = sessions.get(session) or busiest()
        if x is None:
            return {"error": "no live sessions"}
        b = x.bill.as_dict()
        return {"session": x.id, "total": b["total"], "lines": [{k: l.get(k) for k in ("id", "name", "price", "status", "zone",
                "confidence", "note", "seen")} for l in b["lines"]]}

    def t_events(limit: int = 40):
        return list(events_log)[-max(1, min(int(limit), 200)):]

    def t_stock(zone: str = ""):
        v = shop.view()
        sh = [x for x in v["shelves"] if not zone or zone.lower() in (x["name"].lower(), (x.get("zone") or "").lower())]
        names = {p["sku"]: p["name"] for p in v["products"]}
        return [{"shelf": x["name"], "zone": x.get("zone") or x["name"], "state": x["state"], "units": x["units"],
                 "items": [{"product": names.get(i["sku"], i["sku"]), "qty": i["qty"], "min": i["min"]} for i in x["items"]]}
                for x in sh]

    def t_restock():
        alerts = [e for e in events_log if e.get("type") == "restock"]
        low = [x for x in t_stock() if x["state"] in ("low", "empty", "alert")]
        return {"camera_alerts": alerts[-20:], "below_minimum_or_flagged": low}

    def t_decisions():
        return agent_stats()

    def t_layout():
        v = shop.view()
        return {"shop": v["name"], "shelves": [{"name": x["name"], "kind": x["kind"], "zone": x.get("zone") or x["name"],
                "products": [i["sku"] for i in x["items"]]} for x in v["shelves"]]}

    def t_uncertainty():
        per: Dict[str, dict] = {}
        for e in events_log:
            z = e.get("zone") or (e.get("line") or {}).get("zone")
            if not z:
                continue
            d = per.setdefault(z, {"takes": 0, "verify": 0, "disputed": 0, "unchecked": 0, "put_backs": 0,
                                   "low_confidence_examples": []})
            t = e.get("type")
            if t == "take":
                d["takes"] += 1
            elif t == "put":
                d["put_backs"] += 1
            elif t == "unchecked":
                d["unchecked"] += 1
            elif t == "dispute":
                d["disputed"] += 1
            elif t == "line" and (e.get("line") or {}).get("status") == "verify":
                d["verify"] += 1
                if len(d["low_confidence_examples"]) < 3:
                    d["low_confidence_examples"].append(e["line"])
        return per

    if app.state.agents is not None and app.state.agents.assistant is None:
        from .agents import ShopAssistant, tool_schemas
        app.state.agents.assistant = ShopAssistant(app.state.agents.client, {
            "get_sessions": t_sessions, "get_bill": t_bill, "get_events": t_events, "get_stock": t_stock,
            "get_restock_alerts": t_restock, "get_exit_decisions": t_decisions, "get_layout": t_layout,
            "get_uncertainty": t_uncertainty}, tool_schemas())

    @app.post("/v1/sessions/{sid}/bill/{line_id}/review")
    def review(sid: str, line_id: int, body: dict):
        s = get(sid)
        if line_id not in s.bill.lines:
            raise HTTPException(404, "no such line")
        try:
            line = s.bill.review(line_id, body.get("action"), body.get("by", "staff"), body.get("note"), round(s.now(), 1))
        except ValueError as e:
            raise HTTPException(400, str(e))
        ag = app.state.agents
        adv = ag.advisor.cache.get((sid, line_id)) if ag is not None else None
        if adv:                                        # the audit keeps the agent's advice next to the person's decision
            s.bill.audit[-1]["agent"] = {"recommendation": adv["recommendation"], "confidence": adv["confidence"],
                                         "model": adv["model"]}
        rec = adv["recommendation"] if adv else None
        decisions.append({"session": sid, "line": line_id, "name": line.name, "action": body.get("action"),
                          "advice": rec, "agrees": (rec == "keep" and body.get("action") == "approve") or
                          (rec == "remove" and body.get("action") == "remove"), "at": round(time.time(), 1)})
        s.publish([{"type": "review", "line": vars(line).copy(), "bill": s.bill.as_dict()}])
        return s.bill.as_dict()

    async def ws_session(ws: WebSocket, sid: str) -> Optional[Session]:
        if not authorised(ws.headers, ws.query_params):
            await ws.close(code=4401)
            return None
        if sid not in sessions:
            await ws.close(code=4404)
            return None
        await ws.accept()
        return sessions[sid]

    @app.websocket("/v1/sessions/{sid}/events")
    async def events(ws: WebSocket, sid: str):
        s = await ws_session(ws, sid)
        if s is None:
            return
        q: asyncio.Queue = asyncio.Queue()
        s.subscribers.add(q)
        try:
            await ws.send_text(dumps({"type": "bill", "bill": s.bill.as_dict()}))
            while True:
                await ws.send_text(dumps(await q.get()))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            s.subscribers.discard(q)

    @app.websocket("/v1/sessions/{sid}/ingest")
    async def ingest(ws: WebSocket, sid: str):
        s = await ws_session(ws, sid)
        if s is None:
            return
        try:
            while True:
                data = await ws.receive_bytes()
                try:
                    s.offer(decode_jpeg(data))
                except ValueError:
                    await ws.send_json({"type": "error", "message": "frame was not a JPEG"})
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="vjev", choices=["vjev", "nemotron", "stub"])
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--pose-weights", default="yolo11n-pose.pt")
    args = ap.parse_args()
    import uvicorn
    if args.engine == "stub":
        from .models import StubEngine
        engine = StubEngine(lambda t: 0.9 if (t % 8) > 4 else 0.1)    # a "take" every 8 s
    elif args.engine == "nemotron":                              # NVIDIA Nemotron VL via Nebius Token Factory
        from .models import NemotronEngine
        engine = NemotronEngine(args.preset, pose_weights=args.pose_weights)
    else:
        from .models import VjevEngine
        engine = VjevEngine(args.preset, pose_weights=args.pose_weights)
    # behind a reverse proxy (Caddy): trust its X-Forwarded-* headers for scheme and client address
    uvicorn.run(create_app(engine), host=args.host, port=args.port, proxy_headers=True, forwarded_allow_ips="127.0.0.1")


if __name__ == "__main__":
    main()
