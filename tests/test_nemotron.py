"""NemotronEngine with a fake hosted backend: no API key, no network, no GPU."""
from collections import deque

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("vjev")
from vjev.tokenfactory import BackendUnavailable  # noqa: E402

from aisle_agent.logic import Catalogue  # noqa: E402
from aisle_agent.models import ModelUnavailable, NemotronEngine, StubEngine, Track, VjevEngine  # noqa: E402
from aisle_agent.pipeline import Options, Session  # noqa: E402

CAT = [{"sku": "coke", "name": "can of cola", "price": 40}, {"sku": "pepsi", "name": "can of pepsi", "price": 40},
       {"sku": "jam", "name": "jar of jam", "price": 189}]


class FakeClient:
    def __init__(self):
        self.usage = {"calls": 0, "cost_usd": 0.0, "errors": 0}
        self.trace = deque(maxlen=200)


class FakeBackend:
    """Yes/No logits from `holding` (a list of booleans, one per crop); choices favour `pick`."""
    model = "nvidia/Nemotron-3.5-Super-VL"

    def __init__(self, holding=(), pick=None, down=False):
        self.holding, self.pick, self.down = list(holding), pick, down
        self.client, self.logprobs_supported, self.calls = FakeClient(), True, 0

    def batch_next_token_logits(self, items):
        if self.down:
            raise BackendUnavailable("503")
        out = []
        for _ in items:
            h = self.holding.pop(0)
            out.append(np.array([2.0, -2.0] if h else [-2.0, 2.0]))
        return out

    def next_token_logits(self, image, text, tokens):
        self.calls += 1
        if self.down:
            raise BackendUnavailable("503")
        if tokens == ["Yes", "No"]:
            return np.array([2.0, -2.0])
        lines = [l for l in text.splitlines() if ". " in l and l[0].isalpha() and l[1] == "."]
        target = next((l[0] for l in lines if self.pick and l.endswith(self.pick)), tokens[0])
        return np.array([4.0 if t == target else 0.0 for t in tokens])


def engine(**kw):
    return NemotronEngine(backend=FakeBackend(**kw), naming="siglip")


def crops(n):
    return [Image.new("RGB", (64, 64)) for _ in range(n)]


def test_hand_check_is_calibrated_yes_no_over_the_hosted_model():
    e = engine(holding=[False, True])
    pb, pa = e.p_hand(crops(2))
    assert pb < 0.1 and pa > 0.9 and e.label.startswith("Nemotron")


def test_hosted_model_down_raises_model_unavailable():
    e = engine(down=True)
    with pytest.raises(ModelUnavailable):
        e.p_hand(crops(1))
    with pytest.raises(ModelUnavailable):
        e.stock(crops(1))


def test_second_opinion_when_siglip_is_close(monkeypatch):
    frame = np.zeros((200, 200, 3), np.uint8)
    close = {"sku": "coke", "confidence": 0.52, "method": "image-match/siglip",
             "probabilities": {"coke": 0.52, "pepsi": 0.45, "jam": 0.03}, "box": None, "box_confidence": 1.0}
    monkeypatch.setattr(VjevEngine, "identify", lambda self, *a, **k: dict(close))
    cat, region = Catalogue(CAT), (10, 10, 60, 60)
    agree = engine(pick="can of cola").identify(frame, None, cat, region)
    assert agree["second_opinion"]["agrees"] and agree["confidence"] > 0.52 and agree["method"].endswith("+nemotron")
    disagree = engine(pick="can of pepsi").identify(frame, None, cat, region)
    assert disagree["sku"] == "coke" and not disagree["second_opinion"]["agrees"] and disagree["confidence"] <= 0.45
    down = engine(down=True).identify(frame, None, cat, region)
    assert down["second_opinion"]["unavailable"] and down["confidence"] <= 0.45
    clear = dict(close, probabilities={"coke": 0.9, "pepsi": 0.08, "jam": 0.02}, confidence=0.9)
    monkeypatch.setattr(VjevEngine, "identify", lambda self, *a, **k: dict(clear))
    e = engine(pick="can of pepsi")
    assert "second_opinion" not in e.identify(frame, None, cat, region) and e.backend.calls == 0


class DownStub(StubEngine):
    def p_hand(self, crops):
        raise ModelUnavailable("Token Factory timeout")


def test_pipeline_marks_visits_unchecked_when_the_model_is_down():
    eng = DownStub(lambda t: 0.0, sku="coke", hand_fn=lambda t: (0.2, 0.5) if 1.0 <= t < 1.5 else (0.5, 0.5))
    s = Session("z", eng, Catalogue(CAT), Options(), zones=[{"id": "top", "box": [0, 0, 0.42, 1], "skus": ["coke"]}])
    frame = np.zeros((360, 640, 3), np.uint8)
    ev = []
    for i in range(40):
        m, jobs = s.process(frame, i / 10)
        ev += [x for j in jobs for x in s.resolve(j)]
    assert [e["type"] for e in ev] == ["visit", "unchecked"] if ev and ev[0]["type"] == "visit" else ["unchecked"]
    assert s.bill.as_dict()["lines"] == [] and s.counts["unchecked"] == 1


def test_cpu_boxes_name_products_with_nemotron():
    e = NemotronEngine(backend=FakeBackend(pick="jar of jam"), naming="nemotron")
    cat = Catalogue(CAT + [{"sku": "x", "name": "x", "price": 1, "images": ["data:image/png;base64,AAAA"]}])
    e.prepare(cat)                                        # no embedder is loaded
    assert cat.refs == {} and e._embedder is None
    out = e.identify(np.zeros((200, 200, 3), np.uint8), None, Catalogue(CAT), (10, 10, 60, 60))
    assert out["sku"] == "jam" and out["method"].startswith("nemotron-choice")
