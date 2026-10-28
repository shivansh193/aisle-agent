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


def create_app(engine, max_sessions: int = 4, layout=None, agents=None) -> FastAPI:
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
        if request.url.path.startswith("/v1/") and not authorised(request.headers, request.query_params):
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

    @app.get("/v1/sessions/{sid}")
    def stats(sid: str):
        return get(sid).stats()

    @app.delete("/v1/sessions/{sid}", status_code=204)
    async def delete(sid: str):
        s = get(sid)
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

    @app.post("/v1/report")
    def day_report():
        """End-of-day report on Nemotron-3-Ultra: sales by shelf, restock list, uncertain-take hotspots."""
        ag = app.state.agents
        if ag is None:
            raise HTTPException(503, "the report needs a Token Factory key (TOKEN_FACTORY_API_KEY)")
        from .agents import DayReport
        snap = {"bills": [t_bill(sid) for sid in sessions], "events": t_events(200), "stock": t_stock(),
                "restock": t_restock(), "exit_decisions": agent_stats(), "uncertainty": t_uncertainty()}
        try:
            return DayReport(ag.client).write(snap)
        except Exception as e:
            raise HTTPException(503, f"report unavailable: {str(e)[:160]}")

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
