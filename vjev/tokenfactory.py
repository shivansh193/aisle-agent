"""Hosted backend: NVIDIA Nemotron VL through Nebius Token Factory (OpenAI-compatible API).

    from vjev import Decider
    from vjev.tokenfactory import TokenFactoryBackend
    d = Decider(backend=TokenFactoryBackend())          # TOKEN_FACTORY_API_KEY in the environment

It exposes the same `next_token_logits` primitive as the local Qwen backend, so every vjev question
type works unchanged. Two ways to read it, picked per call and reported in `last["method"]`:

  * "logprobs": one answer token with `top_logprobs`; the option tokens' log-probabilities are the
    logits (missing options get a floor below the lowest returned one). Same maths as local logits.
  * "votes": when the endpoint returns no logprobs, `n` samples at temperature 0.7; the vote share
    (with add-half smoothing) becomes log-probabilities. Coarser; calibrate it separately.

Nemotron models reason before answering by default, so the first token would be "The…", not the
answer. Decisions switch reasoning off (`chat_template_kwargs={"enable_thinking": False}`); then the
answer is the first token and its logprobs are meaningful. `thinking=True` keeps it on.

Failures never turn into answers: a timeout or API error after retries raises `BackendUnavailable`,
so the caller can abstain. Token usage and cost are counted in `usage`; recent calls are kept in
`trace` (model, method, latency, tokens) for display.
"""
import base64
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple

import numpy as np

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
VL_MODEL = "nvidia/Nemotron-3.5-Super-VL"
PRICES = {                                       # $ per 1M tokens (input, output); Token Factory catalog, Oct 2026
    "nvidia/Nemotron-3.5-Super-VL": (0.30, 0.90),
    "nvidia/nemotron-3-super-120b-a12b": (0.30, 0.90),
    "nvidia/Nemotron-3_5-Lightning": (0.06, 0.24),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06, 0.24),
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.00, 3.00),
}


class BackendUnavailable(RuntimeError):
    """The hosted model did not answer (timeout, rate limit, server error). Abstain; never guess."""


def api_key() -> Optional[str]:
    """TOKEN_FACTORY_API_KEY (or NEBIUS_API_KEY) from the environment, else from a .env file nearby."""
    for name in ("TOKEN_FACTORY_API_KEY", "NEBIUS_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    if os.environ.get("TOKEN_FACTORY_NO_DOTENV"):      # tests: never pick up a real key
        return None
    for path in (".env", os.path.join("..", ".env")):
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                k, _, v = line.strip().partition("=")
                if k in ("TOKEN_FACTORY_API_KEY", "NEBIUS_API_KEY") and v:
                    return v.strip().strip('"').strip("'")
    return None


def image_data_uri(image, max_side: int = 512, quality: int = 85) -> str:
    """JPEG data URI of a PIL image, downscaled so the long side is at most `max_side`."""
    im = image.convert("RGB")
    if max(im.size) > max_side:
        k = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class TokenFactoryClient:
    """Minimal OpenAI-compatible chat client (stdlib only) with retries, timeouts and cost tracking."""

    def __init__(self, key: Optional[str] = None, base_url: str = BASE_URL, timeout: float = 20.0,
                 retries: int = 2, max_concurrency: int = 8, transport=None):
        self.key = key or api_key()
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout, self.retries = timeout, retries
        self.sem = threading.Semaphore(max_concurrency)
        self.transport = transport or self._http       # tests pass a fake: (path, body) -> dict
        self.lock = threading.Lock()
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "errors": 0}
        self.trace = deque(maxlen=200)

    def _http(self, path: str, body: dict) -> dict:
        if not self.key:
            raise BackendUnavailable("no Token Factory API key (set TOKEN_FACTORY_API_KEY)")
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def chat(self, model: str, messages: list, purpose: str = "", **params) -> dict:
        body = {"model": model, "messages": messages, **params}
        last = None
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                with self.sem:
                    out = self.transport("chat/completions", body)
                self._account(model, out, time.perf_counter() - t0, purpose)
                return out
            except urllib.error.HTTPError as e:
                last = e
                if e.code not in (408, 409, 429, 500, 502, 503, 504):
                    break                                   # a bad request won't get better
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                last = e
            time.sleep(min(4.0, 0.5 * 2 ** attempt))
        with self.lock:
            self.usage["errors"] += 1
            self.trace.append({"t": time.time(), "model": model, "purpose": purpose, "error": str(last)[:120]})
        raise BackendUnavailable(f"{model}: {last}")

    def _account(self, model, out, seconds, purpose):
        u = out.get("usage") or {}
        pin, pout = PRICES.get(model, (0.0, 0.0))
        cost = (u.get("prompt_tokens", 0) * pin + u.get("completion_tokens", 0) * pout) / 1e6
        with self.lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            self.usage["completion_tokens"] += u.get("completion_tokens", 0)
            self.usage["cost_usd"] = round(self.usage["cost_usd"] + cost, 6)
            self.trace.append({"t": time.time(), "model": model, "purpose": purpose, "ms": round(seconds * 1000),
                               "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                               "cost_usd": round(cost, 6)})


def _norm(tok: str) -> str:
    return tok.strip().strip(".").lower()


class TokenFactoryBackend:
    """vjev backend over a hosted VLM. `mode`: "auto" (logprobs, falling back to votes), "logprobs" or "votes"."""

    def __init__(self, model: str = VL_MODEL, client: Optional[TokenFactoryClient] = None, mode: str = "auto",
                 votes: int = 5, vote_temperature: float = 0.7, max_side: int = 512, thinking: bool = False,
                 **client_kw):
        if mode not in ("auto", "logprobs", "votes"):
            raise ValueError("mode must be auto, logprobs or votes")
        self.model, self.mode, self.votes, self.vote_temperature = model, mode, votes, vote_temperature
        self.max_side = max_side
        self.extra = {} if thinking else {"chat_template_kwargs": {"enable_thinking": False}}
        self.client = client or TokenFactoryClient(**client_kw)
        self.logprobs_supported: Optional[bool] = None if mode == "auto" else mode == "logprobs"
        self.last = {}

    # ---------------------------------------------------------------- primitive

    def _messages(self, image, text):
        content = [{"type": "text", "text": text}]
        if image is not None:
            content.insert(0, {"type": "image_url", "image_url": {"url": image_data_uri(image, self.max_side)}})
        return [{"role": "system", "content": "Answer with exactly one token from the allowed answers. No explanation."},
                {"role": "user", "content": content}]

    def next_token_logits(self, image, text: str, tokens: Sequence[str]) -> np.ndarray:
        msgs = self._messages(image, text)
        if self.logprobs_supported is not False:
            out = self.client.chat(self.model, msgs, purpose="decision", max_tokens=1, temperature=0,
                                   logprobs=True, top_logprobs=20, **self.extra)
            z = self._from_logprobs(out, tokens)
            if z is not None:
                self.logprobs_supported = True
                self.last = {"method": "logprobs", "model": self.model}
                return z
            if self.logprobs_supported is True:          # was working; this reply just lacks them
                self.last = {"method": "logprobs-missing", "model": self.model}
                raise BackendUnavailable("logprobs missing from reply")
            self.logprobs_supported = False
        return self._from_votes(msgs, tokens)

    @staticmethod
    def _from_logprobs(out: dict, tokens: Sequence[str]) -> Optional[np.ndarray]:
        try:
            first = out["choices"][0]["logprobs"]["content"][0]
        except (KeyError, IndexError, TypeError):
            return None
        cands = [(first["token"], first["logprob"])] + [(c["token"], c["logprob"]) for c in first.get("top_logprobs") or []]
        best = {}
        for tok, lp in cands:
            k = _norm(tok)
            if k and (k not in best or lp > best[k]):
                best[k] = lp
        if not best:
            return None
        floor = min(best.values()) - 2.0                  # an option outside the top list is less likely than all of it
        return np.array([best.get(_norm(t), floor) for t in tokens], float)

    def _from_votes(self, msgs, tokens: Sequence[str]) -> np.ndarray:
        out = self.client.chat(self.model, msgs, purpose="decision-votes", max_tokens=3,
                               temperature=self.vote_temperature, n=self.votes, **self.extra)
        answers = [_norm((c.get("message") or {}).get("content") or "") for c in out.get("choices", [])]
        if len(answers) < self.votes:                     # endpoint ignored n: top up with single samples
            for _ in range(self.votes - len(answers)):
                o = self.client.chat(self.model, msgs, purpose="decision-votes", max_tokens=3,
                                     temperature=self.vote_temperature, **self.extra)
                answers.append(_norm(o["choices"][0]["message"]["content"] or ""))
        keys = [_norm(t) for t in tokens]
        count = Counter(next((k for k in keys if a == k or a.startswith(k)), None) for a in answers)
        n = len(answers)
        self.last = {"method": "votes", "model": self.model, "votes": dict(count)}
        return np.log(np.array([(count[k] + 0.5) / (n + 0.5 * len(keys)) for k in keys]))

    def batch_next_token_logits(self, items: List[Tuple[object, str, Sequence[str]]]) -> List[np.ndarray]:
        """Concurrent calls (the client's semaphore bounds them). Raises if any call fails."""
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=min(8, len(items))) as ex:
            return list(ex.map(lambda it: self.next_token_logits(*it), items))

    def describe(self, image, prompt: str, max_tokens: int = 60) -> str:
        """A short caption, for evidence explanations."""
        out = self.client.chat(self.model, [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_data_uri(image, self.max_side)}},
            {"type": "text", "text": prompt}]}], purpose="caption", max_tokens=max_tokens, temperature=0, **self.extra)
        return (out["choices"][0]["message"]["content"] or "").strip()

    def ground(self, image, target: str):
        """Box from a JSON reply in 0-1000 normalised coordinates. No token statistics are available."""
        out = self.client.chat(self.model, self._messages(image, (
            f'Locate "{target}". Reply with JSON only: {{"bbox": [x0, y0, x1, y1]}} in 0-1000 coordinates, '
            'or {"bbox": null} if it is not in the image.'))[1:], purpose="ground", max_tokens=40, temperature=0,
            **self.extra)
        text = out["choices"][0]["message"]["content"] or ""
        nums = re.findall(r"-?\d+\.?\d*", text)
        stats = {"mean_p": None, "min_p": None, "mean_entropy": None}
        if len(nums) < 4:
            return None, stats
        x0, y0, x1, y1 = [float(v) / 1000 for v in nums[:4]]
        return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], stats
