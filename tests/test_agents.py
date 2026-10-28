"""Exit advisor and shop assistant with a fake Token Factory client (no network)."""
import json
from collections import deque

import pytest

pytest.importorskip("vjev")
from fastapi.testclient import TestClient  # noqa: E402

from aisle_agent.agents import LIGHTNING, SUPER, ExitAdvisor, ShopAssistant, tool_schemas  # noqa: E402
from aisle_agent.models import StubEngine  # noqa: E402
from aisle_agent.server import Agents, create_app  # noqa: E402

CAT = [{"sku": "a", "name": "packet of noodles", "price": 10}, {"sku": "b", "name": "carton of milk", "price": 68}]


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.usage, self.trace = {"calls": 0, "cost_usd": 0.0, "errors": 0}, deque()

    def chat(self, model, messages, purpose="", **kw):
        self.calls.append({"model": model, "messages": messages, "purpose": purpose, **kw})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def reply(content=None, tool_calls=None):
    return {"choices": [{"message": {"content": content, "tool_calls": tool_calls}}]}


def test_advisor_parses_json_and_never_invents_a_recommendation():
    c = FakeClient([reply('Sure: {"explanation": "Hand went in empty, came out holding.", "recommendation": "keep", "confidence": 0.8}'),
                    reply("no json at all")])
    adv = ExitAdvisor(c, caption=False)
    line = {"name": "carton of milk", "status": "verify", "confidence": 0.41, "zone": "Dairy"}
    ev = {"take": {"t": 3.2, "p_before": 0.1, "p_after": 0.9}, "candidates": {"b": 0.41, "a": 0.38}, "method": "image-match"}
    a = adv.advise(("s", 1), line, ev, ["carton of milk", "packet of noodles"])
    assert a["recommendation"] == "keep" and a["confidence"] == 0.8 and a["model"] == SUPER
    facts = json.loads(c.calls[0]["messages"][1]["content"])
    assert facts["naming"]["margin"] == pytest.approx(0.03) and facts["hand_at_shelf"]["p_holding_after"] == 0.9
    assert adv.advise(("s", 1), line, ev, []) is a                       # cached while the status is unchanged
    b = adv.advise(("s", 2), line, ev, [])
    assert b["recommendation"] == "ask_shopper"                            # unparseable -> the safe choice


def test_assistant_runs_tools_and_picks_the_model():
    call = [{"id": "c1", "type": "function", "function": {"name": "get_stock", "arguments": '{"zone": "Dairy"}'}}]
    c = FakeClient([reply(None, call), reply("Dairy has 2 cartons left, below its minimum of 4.")])
    seen = {}
    a = ShopAssistant(c, {"get_stock": lambda zone="": seen.setdefault("zone", zone) and [{"shelf": "Dairy", "units": 2}]},
                      tool_schemas())
    out = a.ask("What's running low?")
    assert out["model"] == LIGHTNING and out["tools"] == ["get_stock"] and seen["zone"] == "Dairy"
    assert "2 cartons" in out["answer"] and c.calls[1]["messages"][-1]["role"] == "tool"
    assert a.pick_model("Why were shelf B's takes uncertain today?") == SUPER


def test_assistant_falls_back_to_a_data_snapshot_without_tool_support():
    c = FakeClient([RuntimeError("400 tools not supported"), reply("Nothing is low.")])
    a = ShopAssistant(c, {"get_stock": lambda zone="": []}, tool_schemas())
    out = a.ask("What's running low?")
    assert out["mode"] == "snapshot" and a.tool_calling is False and "Shop data" in c.calls[1]["messages"][0]["content"]


def test_exit_flow_records_advice_and_agreement(monkeypatch):
    monkeypatch.delenv("AISLE_KEY", raising=False)
    c = FakeClient([reply('{"explanation": "Too close to call.", "recommendation": "remove", "confidence": 0.7}')])
    with TestClient(create_app(StubEngine(lambda t: 0.9 if t >= 1.0 else 0.1, confidence=0.3), agents=Agents(c))) as api:
        s = api.post("/v1/sessions", json={"catalogue": CAT, "options": {"hold_hz": 10, "async_identify": False}}).json()
        for i in range(30):
            api.post(f"{s['frames_url']}?t={i / 10}", content=_jpeg(), headers={"Content-Type": "image/jpeg"})
        lid = api.get(s["bill_url"]).json()["lines"][0]["id"]
        adv = api.get(f"/v1/sessions/{s['session_id']}/bill/{lid}/advice").json()
        assert adv["recommendation"] == "remove"
        bill = api.post(f"/v1/sessions/{s['session_id']}/bill/{lid}/review", json={"action": "remove"}).json()
        assert bill["audit"][-1]["agent"]["recommendation"] == "remove"
        st = api.get("/v1/agents/stats").json()
        assert st["agreement_rate"] == 1.0 and st["decisions"] == 1
        assert api.get("/v1/models/trace").json()["sources"]


def test_advice_and_assistant_say_so_without_a_key(monkeypatch):
    monkeypatch.delenv("AISLE_KEY", raising=False)
    with TestClient(create_app(StubEngine(lambda t: 0.1))) as api:
        assert api.post("/v1/assistant", json={"question": "hi"}).status_code == 503


def _jpeg():
    import cv2
    import numpy as np
    return cv2.imencode(".jpg", np.zeros((360, 640, 3), np.uint8))[1].tobytes()
