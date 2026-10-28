"""Token Factory backend tests with a fake HTTP transport (no network, no key)."""
import urllib.error

import numpy as np
import pytest
from PIL import Image

from vjev import Calibration, Decider
from vjev.calibration import Logistic
from vjev.tokenfactory import BackendUnavailable, TokenFactoryBackend, TokenFactoryClient, image_data_uri


def img():
    return Image.new("RGB", (800, 400), "white")


def lp_reply(top, usage=(120, 1)):
    tok, lp = top[0]
    return {"choices": [{"message": {"content": tok}, "logprobs": {"content": [
        {"token": tok, "logprob": lp, "top_logprobs": [{"token": t, "logprob": l} for t, l in top]}]}}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}


def votes_reply(answers):
    return {"choices": [{"message": {"content": a}} for a in answers], "usage": {"prompt_tokens": 100, "completion_tokens": len(answers)}}


class Fake:
    def __init__(self, replies):
        self.replies, self.bodies = list(replies), []

    def __call__(self, path, body):
        self.bodies.append(body)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def backend(replies, **kw):
    fake = Fake(replies)
    client = TokenFactoryClient(key="test", transport=fake, retries=kw.pop("retries", 0))
    return TokenFactoryBackend(client=client, **kw), fake


def test_logprobs_become_logits_with_floor_for_missing_options():
    b, fake = backend([lp_reply([("Yes", -0.1), (" no", -2.4), ("Maybe", -5.0)])])
    z = b.next_token_logits(img(), "Holding?", ["Yes", "No"])
    assert np.allclose(z, [-0.1, -2.4]) and b.last["method"] == "logprobs"
    body = fake.bodies[0]
    assert body["logprobs"] is True and body["max_tokens"] == 1 and body["model"] == "nvidia/Nemotron-3.5-Super-VL"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}          # answer first, no reasoning preamble
    assert body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    b2, _ = backend([lp_reply([("A", -0.2), ("B", -1.9)])])
    z = b2.next_token_logits(img(), "Which?", ["A", "B", "C"])
    assert z[2] == pytest.approx(-1.9 - 2.0)                      # C not in the top list -> below everything returned


def test_falls_back_to_votes_when_no_logprobs():
    no_lp = {"choices": [{"message": {"content": "Yes"}}], "usage": {"prompt_tokens": 100, "completion_tokens": 1}}
    b, fake = backend([no_lp, votes_reply(["Yes", "yes.", "No", "Yes", "Yes"])])
    z = b.next_token_logits(img(), "Holding?", ["Yes", "No"])
    p = np.exp(z) / np.exp(z).sum()
    assert b.last["method"] == "votes" and b.logprobs_supported is False
    assert p[0] == pytest.approx(4.5 / 5.5 / (4.5 / 5.5 + 1.5 / 5.5))
    assert fake.bodies[1]["n"] == 5 and fake.bodies[1]["temperature"] == 0.7
    b.client.transport = Fake([votes_reply(["No"] * 5)])           # later calls go straight to votes
    b.next_token_logits(img(), "Holding?", ["Yes", "No"])
    assert "logprobs" not in b.client.transport.bodies[0]


def test_votes_top_up_when_n_is_ignored():
    b, fake = backend([votes_reply(["Yes"]), votes_reply(["No"]), votes_reply(["Yes"])], mode="votes", votes=3)
    b.next_token_logits(img(), "Holding?", ["Yes", "No"])
    assert len(fake.bodies) == 3 and b.last["votes"] == {"yes": 2, "no": 1}


def test_failures_raise_instead_of_guessing_and_are_counted():
    err = urllib.error.HTTPError("u", 503, "busy", {}, None)
    b, fake = backend([err, err, err], retries=2)
    with pytest.raises(BackendUnavailable):
        b.next_token_logits(img(), "Holding?", ["Yes", "No"])
    assert len(fake.bodies) == 3 and b.client.usage["errors"] == 1
    b2, fake2 = backend([urllib.error.HTTPError("u", 400, "bad", {}, None)], retries=2)
    with pytest.raises(BackendUnavailable):
        b2.next_token_logits(img(), "Holding?", ["Yes", "No"])
    assert len(fake2.bodies) == 1                                   # a 400 is not retried


def test_usage_cost_and_trace():
    b, _ = backend([lp_reply([("Yes", -0.1)], usage=(1_000_000, 0))])
    b.next_token_logits(img(), "Holding?", ["Yes", "No"])
    u = b.client.usage
    assert u["calls"] == 1 and u["cost_usd"] == pytest.approx(0.30)
    t = b.client.trace[-1]
    assert t["model"] == "nvidia/Nemotron-3.5-Super-VL" and t["purpose"] == "decision" and "ms" in t


def test_decider_runs_on_the_hosted_backend():
    b, _ = backend([lp_reply([("Yes", -0.05), ("No", -3.0)]), lp_reply([("2", -0.3), ("1", -1.5)])])
    cal = Calibration()
    cal.noul_platt = Logistic([0.0], [1.0], [1.0, 0.0])
    d = Decider(backend=b, calibration=cal)
    out = d.decide({"assets": {"i": img()}, "questions": {
        "h": {"type": "noul", "question": "Is the hand holding a product?", "image": "i"},
        "s": {"type": "score", "question": "How full?", "levels": ["empty", "low", "full"], "image": "i"}}})
    assert out["answers"]["h"]["answer"] is True and out["answers"]["s"]["level"] == "low"


def test_batch_and_image_downscale():
    b, _ = backend([lp_reply([("Yes", -0.1)]), lp_reply([("No", -0.1)])])
    zs = b.batch_next_token_logits([(img(), "a", ["Yes", "No"]), (img(), "b", ["Yes", "No"])])
    assert len(zs) == 2
    from PIL import Image as I
    import base64, io
    uri = image_data_uri(Image.new("RGB", (2000, 1000)), max_side=512)
    assert I.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).size == (512, 256)
