import asyncio

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from aisle_agent.logic import Bill, Catalogue, HandVisits, TakeDetector, Zone, classify_visit
from aisle_agent.models import StubEngine
from aisle_agent.pipeline import Options, Session
from aisle_agent.server import create_app

CAT = [{"sku": "a", "name": "packet of biscuits", "price": 10}, {"sku": "b", "name": "carton of milk", "price": 68}]


def feed(det, ps, hz=10):
    return [r for i, p in enumerate(ps) if (r := det.update(i / hz, p)) is not None]


def test_take_fires_once_after_confirm():
    ps = [0.1] * 10 + [0.95] * 20
    rises = feed(TakeDetector(), ps)
    assert len(rises) == 1 and 1.0 <= rises[0] <= 1.4


def test_blip_is_not_a_take():
    assert feed(TakeDetector(), [0.1] * 10 + [0.95] * 2 + [0.05] * 10) == []


def test_already_holding_at_start_is_not_a_take():
    assert feed(TakeDetector(), [0.95] * 30) == []


def test_second_take_after_release():
    ps = [0.1] * 10 + [0.95] * 15 + [0.05] * 15 + [0.95] * 15
    assert len(feed(TakeDetector(), ps)) == 2


def test_bill_flags_low_confidence_and_dispute_keeps_line():
    cat = Catalogue(CAT)
    bill = Bill(verify_below=0.5)
    a = bill.add(cat.by_sku["a"], 0.9, 1, 1.0)
    b = bill.add(cat.by_sku["b"], 0.3, 1, 2.0)
    assert (a.status, b.status) == ("ok", "verify")
    bill.dispute(a.id)
    d = bill.as_dict()
    assert d["total"] == 78 and len(d["lines"]) == 2 and set(d["to_verify"]) == {a.id, b.id}



def test_repeat_take_merges_into_one_line():
    cat = Catalogue(CAT)
    bill = Bill(verify_below=0.5, merge_within=6.0)
    a = bill.add(cat.by_sku["a"], 0.4, 1, 1.0, zone="z")
    again = bill.add(cat.by_sku["a"], 0.8, 1, 4.0, zone="z")          # same hand re-grabbing
    assert again is a and a.seen == 2 and a.status == "ok" and a.confidence == 0.8
    other = bill.add(cat.by_sku["a"], 0.8, 2, 4.5, zone="z")          # another shopper: its own line
    late = bill.add(cat.by_sku["a"], 0.8, 1, 20.0, zone="z")          # long after: a second item
    assert len({a.id, other.id, late.id}) == 3
    bill.dispute(late.id)
    assert bill.add(cat.by_sku["a"], 0.9, 1, 21.0, zone="z").id != late.id   # a disputed line is never reopened

def test_catalogue_validation():
    with pytest.raises(ValueError):
        Catalogue([])
    with pytest.raises(ValueError):
        Catalogue([CAT[0], CAT[0]])


def test_session_take_then_line():
    eng = StubEngine(lambda t: 0.9 if t >= 1.0 else 0.1, sku="b", confidence=0.3)
    s = Session("x", eng, Catalogue(CAT), Options(hold_hz=10))
    frame = np.zeros((360, 640, 3), np.uint8)
    msgs = []
    for i in range(40):
        m, jobs = s.process(frame, i / 10)
        msgs += m + [x for j in jobs for x in s.resolve(j)]
    takes = [m for m in msgs if m["type"] == "take"]
    lines = [m for m in msgs if m["type"] == "line"]
    assert len(takes) == 1 and len(lines) == 1
    line = lines[0]["line"]
    assert line["sku"] == "b" and line["status"] == "verify"
    assert 0.5 <= lines[0]["latency_s"] <= 1.2
    assert all(0 <= v <= 1 for v in line["box"])


def jpeg(frame=None):
    frame = np.zeros((360, 640, 3), np.uint8) if frame is None else frame
    return cv2.imencode(".jpg", frame)[1].tobytes()


def test_api_frames_bill_dispute_and_events(monkeypatch):
    monkeypatch.delenv("AISLE_KEY", raising=False)
    eng = StubEngine(lambda t: 0.9 if t >= 1.0 else 0.1)
    with TestClient(create_app(eng)) as c:
        assert c.post("/v1/sessions", json={"catalogue": []}).status_code == 400
        s = c.post("/v1/sessions", json={"catalogue": CAT, "options": {"hold_hz": 10, "async_identify": False}}).json()
        with c.websocket_connect(s["events_ws"]) as ws:
            assert ws.receive_json()["type"] == "bill"
            for i in range(30):
                r = c.post(f"{s['frames_url']}?t={i / 10}", content=jpeg(), headers={"Content-Type": "image/jpeg"})
                assert r.status_code == 200
            seen = [ws.receive_json()["type"] for _ in range(32)]
        assert "take" in seen and "line" in seen
        bill = c.get(s["bill_url"]).json()
        assert bill["total"] == 10 and bill["lines"][0]["status"] == "ok"
        lid = bill["lines"][0]["id"]
        bill = c.post(f"/v1/sessions/{s['session_id']}/bill/{lid}/dispute").json()
        assert bill["lines"][0]["status"] == "disputed" and bill["total"] == 10
        assert c.post(s["frames_url"], content=b"nope").status_code == 400
        st = c.get(f"/v1/sessions/{s['session_id']}").json()
        assert st["counts"]["processed"] == 30 and "hold_ms" in st["timing_ms"]
        assert c.delete(f"/v1/sessions/{s['session_id']}").status_code == 204
        assert c.get(s["bill_url"]).status_code == 404



def test_shopper_link_needs_no_key_and_only_disputes(monkeypatch):
    monkeypatch.setenv("AISLE_KEY", "k1")
    auth = {"Authorization": "Bearer k1"}
    eng = StubEngine(lambda t: 0.9 if t >= 1.0 else 0.1)
    with TestClient(create_app(eng)) as c:
        s = c.post("/v1/sessions", json={"catalogue": CAT, "options": {"hold_hz": 10, "async_identify": False}},
                   headers=auth).json()
        url = s["shopper_url"]
        tok = url.rsplit("/", 1)[1]
        assert c.get(url).status_code == 200 and c.get("/b/nope").status_code == 404
        with c.websocket_connect(f"/p/{tok}/events") as ws:
            assert ws.receive_json()["lines"] == []
            for i in range(30):
                c.post(f"{s['frames_url']}?t={i / 10}", content=jpeg(), headers={**auth, "Content-Type": "image/jpeg"})
            b = ws.receive_json()
            while not b["lines"]:
                b = ws.receive_json()
        assert set(b["lines"][0]) <= {"id", "name", "price", "status", "note", "seen", "zone", "t"}   # no internals
        lid = b["lines"][0]["id"]
        b = c.post(f"/p/{tok}/lines/{lid}/not-mine").json()
        assert b["lines"][0]["status"] == "disputed" and b["to_verify"] == 1
        assert c.post(f"/p/{tok}/lines/{lid}/not-mine").status_code == 409
        assert c.post(f"/v1/sessions/{s['session_id']}/bill/{lid}/review", json={"action": "remove"}).status_code == 401
        assert c.delete(f"/v1/sessions/{s['session_id']}", headers=auth).status_code == 204
        assert c.get(f"/p/{tok}/bill").status_code == 404

def test_api_key(monkeypatch):
    monkeypatch.setenv("AISLE_KEY", "k1")
    with TestClient(create_app(StubEngine(lambda t: 0.1))) as c:
        assert c.post("/v1/sessions", json={"catalogue": CAT}).status_code == 401
        assert c.post("/v1/sessions?key=k1", json={"catalogue": CAT}).status_code == 201
        assert c.post("/v1/sessions", json={"catalogue": CAT}, headers={"Authorization": "Bearer k1"}).status_code == 201
        assert c.get("/").status_code == 200


def test_ws_ingest_drops_to_newest(monkeypatch):
    monkeypatch.delenv("AISLE_KEY", raising=False)
    with TestClient(create_app(StubEngine(lambda t: 0.1))) as c:
        s = c.post("/v1/sessions", json={"catalogue": CAT}).json()
        with c.websocket_connect(s["ingest_ws"]) as ws:
            for _ in range(5):
                ws.send_bytes(jpeg())
        import time
        time.sleep(0.5)
        st = c.get(f"/v1/sessions/{s['session_id']}").json()
        assert st["counts"]["received"] == 5 and st["counts"]["processed"] >= 1


def test_async_identify_does_not_block_frames(monkeypatch):
    monkeypatch.delenv("AISLE_KEY", raising=False)
    import time
    eng = StubEngine(lambda t: 0.9 if t >= 1.0 else 0.1, identify_s=1.0)
    with TestClient(create_app(eng)) as c:
        s = c.post("/v1/sessions", json={"catalogue": CAT, "options": {"hold_hz": 10}}).json()
        with c.websocket_connect(s["events_ws"]) as ws:
            ws.receive_json()
            slowest = 0.0
            for i in range(25):
                t0 = time.perf_counter()
                r = c.post(f"{s['frames_url']}?t={i / 10}", content=jpeg(), headers={"Content-Type": "image/jpeg"})
                slowest = max(slowest, time.perf_counter() - t0)
                assert all(e["type"] != "line" for e in r.json()["events"])
            assert slowest < 0.6                      # identification (1 s) ran in the background
            types = []
            while "line" not in types:
                types.append(ws.receive_json()["type"])
        assert c.get(s["bill_url"]).json()["total"] == 10


def test_hand_visits_enter_exit_and_lost():
    v = HandVisits([Zone("z", [0.0, 0.0, 0.3, 1.0])], min_dwell=0.15)
    out = []
    for i, x in enumerate([0.5, 0.4, 0.2, 0.2, 0.2, 0.5, 0.5, 0.2, 0.2, 0.2]):
        out += v.update(i / 10, {(1, 1): (x, 0.5)})
    assert len(out) == 1 and out[0]["zone"] == "z" and out[0]["t_enter"] == 0.2 and out[0]["t_exit"] == 0.5
    out = []
    for i in range(10, 30):
        out += v.update(i / 10, {})                    # hand lost while inside (entered at 0.7)
    assert len(out) == 1 and out[0].get("lost")
    v2 = HandVisits([Zone("z", [0.0, 0.0, 0.3, 1.0])], min_dwell=0.15)
    assert v2.update(0.0, {(1, 1): (0.5, 0.5)}) == [] and v2.update(0.1, {(1, 1): (0.2, 0.5)}) == []
    assert v2.update(0.15, {(1, 1): (0.5, 0.5)}) == []    # too short to count


def test_classify_visit():
    assert classify_visit(0.1, 0.9) == "take"
    assert classify_visit(None, 0.9) == "take"
    assert classify_visit(0.9, 0.1) == "put"
    assert classify_visit(0.9, 0.9) == "touch"
    assert classify_visit(0.1, 0.2) == "touch"


def run_zone_session(hold_fn, hand_fn, zones, cat=CAT, seconds=4.0):
    eng = StubEngine(hold_fn, sku="b", hand_fn=hand_fn)
    s = Session("z", eng, Catalogue(cat), Options(), zones=zones)
    frame = np.zeros((360, 640, 3), np.uint8)
    msgs = []
    for i in range(int(seconds * 10)):
        m, jobs = s.process(frame, i / 10)
        msgs += m + [x for j in jobs for x in s.resolve(j)]
    return s, [m for m in msgs if m["type"] not in ("state", "stock")]


def test_zone_take_then_put_back_flags_line():
    zones = [{"id": "top", "box": [0.0, 0.0, 0.42, 1.0], "skus": ["b"]}]   # resting hand (x 0.5) is "near"
    hand = lambda t: (0.2, 0.5) if 1.0 <= t < 1.5 or 2.5 <= t < 3.0 else (0.5, 0.5)   # two shelf visits
    hold = lambda t: 1.0 if 1.4 <= t < 2.6 else 0.0                                    # holding in between
    s, ev = run_zone_session(hold, hand, zones)
    kinds = [e["type"] for e in ev]
    assert kinds.count("visit") == 2 and "take" in kinds and "put" in kinds
    line = s.bill.as_dict()["lines"][0]
    assert line["sku"] == "b" and line["zone"] == "top" and line["status"] == "verify" and line["note"] == "put-back seen"


def test_zone_touch_adds_nothing():
    zones = [{"id": "top", "box": [0.0, 0.0, 0.3, 1.0], "skus": []}]
    s, ev = run_zone_session(lambda t: 0.0, lambda t: (0.2, 0.5) if 1.0 <= t < 1.5 else (0.5, 0.5), zones)
    assert [e["type"] for e in ev] == ["visit", "touch"] and s.bill.as_dict()["lines"] == []


def test_zone_validation_and_put_endpoint(monkeypatch):
    monkeypatch.delenv("AISLE_KEY", raising=False)
    with TestClient(create_app(StubEngine(lambda t: 0.1))) as c:
        bad = {"catalogue": CAT, "zones": [{"id": "z", "box": [0.5, 0, 0.2, 1], "skus": []}]}
        assert c.post("/v1/sessions", json=bad).status_code == 400
        bad["zones"][0].update(box=[0, 0, 0.3, 1], skus=["nope"])
        assert c.post("/v1/sessions", json=bad).status_code == 400
        s = c.post("/v1/sessions", json={"catalogue": CAT}).json()
        assert s["mode"] == "hold"
        r = c.put(f"/v1/sessions/{s['session_id']}/zones", json=[{"id": "z", "box": [0, 0, 0.3, 1], "skus": ["a"]}])
        assert r.json() == {"mode": "zones", "zones": 1}


def test_rank_products_and_subset_refs():
    from aisle_agent.logic import rank_products
    refs = {"a": np.array([[1.0, 0.0]]), "b": np.array([[0.0, 1.0], [0.6, 0.8]])}
    sku, conf, probs = rank_products(np.array([0.0, 1.0]), refs, ["a", "b"], 0.05)
    assert sku == "b" and conf > 0.99 and set(probs) == {"a", "b"}
    cat = Catalogue(CAT)
    cat.refs = refs
    sub = cat.subset(["b"])
    assert len(sub) == 1 and list(sub.refs) == ["b"]
    with pytest.raises(ValueError):
        rank_products(np.array([1.0, 0.0]), {}, ["a"], 0.05)


def test_decode_image_data_uri_and_url_off(monkeypatch):
    import base64
    from aisle_agent.models import decode_image
    uri = "data:image/jpeg;base64," + base64.b64encode(jpeg()).decode()
    assert decode_image(uri).size == (640, 360)
    monkeypatch.delenv("AISLE_FETCH_URLS", raising=False)
    with pytest.raises(ValueError):
        decode_image("https://example.com/x.jpg")


def test_engine_prepare_called_and_bad_photo_rejected(monkeypatch):
    monkeypatch.delenv("AISLE_KEY", raising=False)
    from aisle_agent.models import decode_image

    class Prep(StubEngine):
        def prepare(self, catalogue):
            catalogue.refs = {p.sku: [decode_image(i) for i in p.images] for p in catalogue.products if p.images}

    with TestClient(create_app(Prep(lambda t: 0.1))) as c:
        bad = [dict(CAT[0], images=["not base64 !!"])] + CAT[1:]
        assert c.post("/v1/sessions", json={"catalogue": bad}).status_code == 400


def test_review_and_audit():
    cat = Catalogue(CAT)
    bill = Bill(verify_below=0.5)
    a = bill.add(cat.by_sku["a"], 0.3, 1, 1.0)
    b = bill.add(cat.by_sku["b"], 0.9, 1, 2.0)
    bill.dispute(b.id)
    assert bill.as_dict()["to_verify"] == [a.id, b.id]
    bill.review(a.id, "approve", "cashier", "checked the bag")
    bill.review(b.id, "remove", "cashier")
    d = bill.as_dict()
    assert d["total"] == 10 and d["to_verify"] == [] and len(d["lines"]) == 2       # removed line stays, out of total
    assert [x["action"] for x in d["audit"]] == ["approve", "remove"] and d["audit"][1]["from"] == "disputed"
    with pytest.raises(ValueError):
        bill.review(a.id, "delete")
    assert bill.flag_put_back(1, "z", ["b"]) is None                                  # a removed line is not re-flagged


def test_exit_check_api_evidence_and_review(monkeypatch):
    monkeypatch.delenv("AISLE_KEY", raising=False)
    zones = [{"id": "top", "box": [0.0, 0.0, 0.42, 1.0], "skus": ["b"]}]
    eng = StubEngine(lambda t: 1.0 if 1.4 <= t < 2.6 else 0.0, sku="b", confidence=0.4,
                     hand_fn=lambda t: (0.2, 0.5) if 1.0 <= t < 1.5 or 2.5 <= t < 3.0 else (0.5, 0.5))
    with TestClient(create_app(eng)) as c:
        s = c.post("/v1/sessions", json={"catalogue": CAT, "zones": zones, "options": {"async_identify": False}}).json()
        for i in range(40):
            c.post(f"{s['frames_url']}?t={i / 10}", content=jpeg(), headers={"Content-Type": "image/jpeg"})
        lst = c.get("/v1/sessions").json()["sessions"]
        assert lst[0]["session_id"] == s["session_id"] and lst[0]["to_verify"] == 1
        bill = c.get(s["bill_url"]).json()
        lid = bill["lines"][0]["id"]
        ev = c.get(f"/v1/sessions/{s['session_id']}/bill/{lid}/evidence").json()
        assert ev["take"]["after"].startswith("data:image/jpeg;base64,") and ev["take"]["p_after"] == 1.0
        assert ev["put_back"]["p_before"] == 1.0 and ev["put_back"]["p_after"] == 0.0
        assert ev["line"]["note"] == "put-back seen"
        r = c.post(f"/v1/sessions/{s['session_id']}/bill/{lid}/review", json={"action": "remove", "note": "back on shelf"})
        assert r.json()["total"] == 0 and r.json()["audit"][0]["note"] == "back on shelf"
        assert c.post(f"/v1/sessions/{s['session_id']}/bill/{lid}/review", json={"action": "x"}).status_code == 400
        assert c.get("/exit").status_code == 200


def test_stock_watch_hysteresis():
    from aisle_agent.logic import StockWatch
    w = StockWatch(low=0.4, ok=0.55, confirm=2)
    full, low = {"empty": 0, "low": 0, "half full": 0, "full": 1.0}, {"empty": 0.2, "low": 0.8, "half full": 0, "full": 0}
    kinds = [e["type"] for d in (full, low, low, low, full) for e in w.update("z", 0, d)]
    assert kinds == ["stock", "stock", "stock", "restock", "stock", "stock", "restocked"]


def test_shelf_watch_in_session_and_api(monkeypatch):
    monkeypatch.delenv("AISLE_KEY", raising=False)
    eng = StubEngine(lambda t: 0.0, stock_fn=lambda t: "full" if t < 4 else "empty")
    with TestClient(create_app(eng)) as c:
        s = c.post("/v1/sessions", json={"catalogue": CAT, "zones": [{"id": "top", "box": [0, 0, 0.3, 1], "skus": []}],
                                         "options": {"async_identify": False, "watch_every": 2.0}}).json()
        seen = []
        for i in range(0, 120, 5):
            r = c.post(f"{s['frames_url']}?t={i / 10}", content=jpeg(), headers={"Content-Type": "image/jpeg"}).json()
            seen += [e["type"] for e in r["events"] if e["type"] in ("stock", "restock")]
        assert seen.count("restock") == 1 and seen.index("restock") > 2
        sh = c.get(f"/v1/sessions/{s['session_id']}/shelves").json()
        assert sh["alerts"] == ["top"] and sh["zones"]["top"]["label"] == "empty"


def test_live_feed_preview(monkeypatch):
    monkeypatch.delenv("AISLE_KEY", raising=False)
    with TestClient(create_app(StubEngine(lambda t: 0.1))) as c:
        s = c.post("/v1/sessions", json={"catalogue": CAT}).json()
        assert c.get(f"/v1/sessions/{s['session_id']}/preview.jpg").status_code == 404
        c.post(f"{s['frames_url']}?t=0", content=jpeg(np.full((720, 1280, 3), 90, np.uint8)), headers={"Content-Type": "image/jpeg"})
        r = c.get(f"/v1/sessions/{s['session_id']}/preview.jpg")
        assert r.status_code == 200 and cv2.imdecode(np.frombuffer(r.content, np.uint8), 1).shape == (540, 960, 3)
        with c.stream("GET", f"/v1/sessions/{s['session_id']}/preview.mjpg?frames=1") as r:
            body = b"".join(r.iter_bytes())
        assert r.headers["content-type"].startswith("multipart/x-mixed-replace") and b"Content-Type: image/jpeg" in body
