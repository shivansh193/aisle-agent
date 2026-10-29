"""Shelf audit: wrong-shelf products and price labels, scheduling, findings, resolution."""
import numpy as np
from fastapi.testclient import TestClient

from aisle_agent.layout import Layout
from aisle_agent.logic import Catalogue, Product, match_label
from aisle_agent.models import StubEngine
from aisle_agent.pipeline import Options, Session
from aisle_agent.server import create_app

CAT = [{"sku": "frozen", "name": "frozen food pack", "price": 220}, {"sku": "milk", "name": "carton of milk", "price": 68}]
ZONES = [{"id": "Freezer", "box": [0.0, 0.0, 0.5, 1.0], "skus": ["frozen", "milk"]}]


def found(p=0.9, labels=()):
    return lambda t: {"misplaced": {"p": p, "item": "ice-cream tub"}, "labels": list(labels), "model": "stub"}


def run(eng, opt, seconds=1.0, request=False):
    s = Session("a", eng, Catalogue(CAT), opt, zones=ZONES)
    frame = np.zeros((360, 640, 3), np.uint8)
    ev = []
    for i in range(int(seconds * 10)):
        if request and i == 2:
            s.request_audit()
        _, jobs = s.process(frame, i / 10)
        ev += [m for j in jobs for m in s.resolve(j)]
    return s, ev


def test_match_label_by_words_or_single_product():
    ps = [Product("frozen", "frozen food pack", 220), Product("milk", "carton of milk", 68)]
    assert match_label("Frozen Pack 199", ps).sku == "frozen"
    assert match_label("MILK 1L", ps).sku == "milk"
    assert match_label("something else", ps) is None
    assert match_label("", ps[:1]).sku == "frozen"


def test_audit_finds_wrong_shelf_and_price_mismatch():
    eng = StubEngine(lambda t: 0.0, hand_fn=lambda t: (0.9, 0.5))       # hand never in the zone
    eng.audit_fn = found(0.9, [{"text": "Frozen pack", "price": 199}, {"text": "Milk", "price": 68}])
    s, ev = run(eng, Options(watch_every=0, audit_every=0.5))         # periodic: first audit one interval in
    kinds = [e["type"] for e in ev]
    assert kinds.count("audit") == 1 and "misplaced" in kinds and kinds.count("price_mismatch") == 1   # milk matches
    pm = next(e for e in ev if e["type"] == "price_mismatch")
    assert pm["sku"] == "frozen" and pm["label_price"] == 199 and pm["catalogue_price"] == 220
    assert s.audits["Freezer"]["crop"].startswith("data:image/jpeg")


def test_audit_runs_on_request_only_when_periodic_is_off_and_low_p_is_not_a_finding():
    eng = StubEngine(lambda t: 0.0, hand_fn=lambda t: (0.9, 0.5))
    eng.audit_fn = found(0.3)
    _, ev = run(eng, Options(watch_every=0, audit_every=0))
    assert ev == []
    _, ev = run(eng, Options(watch_every=0, audit_every=0), request=True)
    assert [e["type"] for e in ev] == ["audit"] and ev[0]["findings"] == []


def test_api_lists_and_resolves_findings_and_updates_price(monkeypatch, tmp_path):
    monkeypatch.delenv("AISLE_KEY", raising=False)
    eng = StubEngine(lambda t: 0.0, hand_fn=lambda t: (0.9, 0.5))
    eng.audit_fn = found(0.9, [{"text": "Frozen pack", "price": 199}])
    lay = Layout(str(tmp_path / "l.json"))
    lay.replace({"name": "S", "products": CAT, "shelves": [{"name": "Freezer", "kind": "freezer", "x": 1, "y": 1, "w": 6, "h": 2,
                                                            "items": [{"sku": "frozen", "qty": 5, "min": 1}]}]})
    with TestClient(create_app(eng, layout=lay)) as api:
        s = api.post("/v1/sessions", json={"catalogue": CAT, "zones": ZONES,
                                           "options": {"async_identify": False, "watch_every": 0, "audit_every": 0}}).json()
        assert api.post(f"/v1/sessions/{s['session_id']}/audit").json()["queued"] == ["Freezer"]
        for i in range(5):
            api.post(f"{s['frames_url']}?t={i / 10}", content=_jpeg(), headers={"Content-Type": "image/jpeg"})
        a = api.get("/v1/audit").json()
        assert a["open"] == 2 and {f["kind"] for f in a["audits"][0]["findings"]} == {"misplaced", "price"}
        assert lay.view()["shelves"][0]["findings"]
        price = next(f for f in a["audits"][0]["findings"] if f["kind"] == "price")
        a = api.post("/v1/audit/resolve", json={"session": s["session_id"], "zone": "Freezer", "idx": price["idx"],
                                                "action": "use_label_price"}).json()
        assert a["open"] == 1
        assert next(p for p in lay.view()["products"] if p["sku"] == "frozen")["price"] == 199
        assert api.post("/v1/audit/resolve", json={"session": s["session_id"], "zone": "Freezer", "idx": 0,
                                                   "action": "nope"}).status_code == 400


def _jpeg():
    import cv2
    return cv2.imencode(".jpg", np.zeros((360, 640, 3), np.uint8))[1].tobytes()
