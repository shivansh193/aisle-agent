"""Text agents on NVIDIA Nemotron via Nebius Token Factory.

ExitAdvisor (Nemotron-3-Super): for a flagged line, reads the evidence (before/after hand crops as
  Nemotron VL captions, the hold scores, the naming candidates and margin, the zone's products, any
  put-back, timestamps) and writes a plain-English explanation plus keep / remove / ask_shopper with a
  confidence. It only recommends: a person decides every line, and the audit log keeps both.

ShopAssistant (Nemotron-3.5-Lightning for lookups, Nemotron-3-Super for multi-step questions):
  a tool-calling agent over the server's own data, read-only.
"""
import json
import re
import threading
from typing import Callable, Dict, List, Optional

SUPER = "nvidia/nemotron-3-super-120b-a12b"
LIGHTNING = "nvidia/Nemotron-3_5-Lightning"
VL = "nvidia/Nemotron-3.5-Super-VL"
RECOMMENDATIONS = ("keep", "remove", "ask_shopper")

EXIT_SYSTEM = """You help a shop's staff settle flagged lines on a shopper's bill at the exit.
A camera system watched a shelf and decided a hand took a product. You get the evidence for one line.
Explain in 1-3 short sentences what the evidence shows, with times and numbers, and say what makes it
uncertain. Then recommend one of: keep (the shopper took it), remove (they didn't), ask_shopper (the
evidence can't settle it). You only advise; staff decide. Never invent evidence that isn't given.
Reply with JSON only: {"explanation": "...", "recommendation": "keep|remove|ask_shopper", "confidence": 0.0-1.0}"""

ASSISTANT_SYSTEM = """You are the shop owner's assistant for Aisle Agent, a camera-based shelf billing system.
Answer questions about the shop using the tools; they return live data. Be brief and concrete: name
shelves, products, counts and times. If the data doesn't answer the question, say what's missing.
Don't make up numbers. Prices are in rupees."""

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}   # answers come first; no reasoning preamble
MULTI_STEP = re.compile(r"\b(why|summar|compare|which .* most|trend|explain|report|cause|recommend|should)\b", re.I)


def _json_from(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class ExitAdvisor:
    def __init__(self, client, vl_backend=None, model: str = SUPER, caption: bool = True):
        self.client, self.vl, self.model, self.caption = client, vl_backend, model, caption
        self.cache: Dict[tuple, dict] = {}
        self.lock = threading.Lock()

    def _captions(self, evidence: dict) -> dict:
        """Nemotron VL describes the before/after hand crops (data URIs) in a few words each."""
        out = {}
        if not (self.caption and self.vl):
            return out
        from vjev.backend import load_image
        for kind in ("take", "put_back"):
            proof = evidence.get(kind) or {}
            for when in ("before", "after"):
                if proof.get(when):
                    out[f"{kind}_{when}"] = self.vl.describe(load_image(proof[when]),
                                                             "In under 15 words: what is this hand holding, if anything?",
                                                             max_tokens=40)
        return out

    def facts(self, line: dict, evidence: dict, zone_products: List[str]) -> dict:
        probs = evidence.get("candidates") or {}
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        take, put = evidence.get("take") or {}, evidence.get("put_back") or {}
        return {"line": {k: line.get(k) for k in ("name", "price", "status", "note", "confidence", "zone", "t", "seen")},
                "naming": {"method": evidence.get("method"), "top3": ranked,
                           "margin": round(ranked[0][1] - ranked[1][1], 3) if len(ranked) > 1 else None,
                           "second_opinion": evidence.get("second_opinion")},
                "hand_at_shelf": {"t": take.get("t"), "p_holding_before": take.get("p_before"),
                                  "p_holding_after": take.get("p_after")},
                "put_back": {"t": put.get("t"), "p_holding_before": put.get("p_before"),
                             "p_holding_after": put.get("p_after")} if put else None,
                "zone_products": zone_products}

    def advise(self, key: tuple, line: dict, evidence: dict, zone_products: List[str]) -> dict:
        with self.lock:
            if key in self.cache and self.cache[key].get("status") == line.get("status"):
                return self.cache[key]
        facts = self.facts(line, evidence, zone_products)
        facts["what_the_crops_show"] = self._captions(evidence)
        out = self.client.chat(self.model, [{"role": "system", "content": EXIT_SYSTEM},
                                            {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}],
                               purpose="exit-advice", max_tokens=300, temperature=0.2, **NO_THINK)
        msg = out["choices"][0]["message"]
        d = _json_from(msg.get("content") or "") or {}
        rec = d.get("recommendation") if d.get("recommendation") in RECOMMENDATIONS else "ask_shopper"
        try:
            conf = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        advice = {"explanation": (d.get("explanation") or msg.get("content") or "").strip()[:600],
                  "recommendation": rec, "confidence": round(conf, 2), "model": self.model,
                  "captions": facts["what_the_crops_show"], "status": line.get("status")}
        with self.lock:
            self.cache[key] = advice
        return advice


class ShopAssistant:
    def __init__(self, client, tools: Dict[str, Callable[..., object]], schemas: List[dict],
                 fast: str = LIGHTNING, deep: str = SUPER, max_steps: int = 5):
        self.client, self.tools, self.schemas = client, tools, schemas
        self.fast, self.deep, self.max_steps = fast, deep, max_steps
        self.tool_calling: Optional[bool] = None

    def pick_model(self, question: str) -> str:
        return self.deep if MULTI_STEP.search(question) or len(question) > 140 else self.fast

    def _run(self, name: str, args: dict):
        fn = self.tools.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}

    def ask(self, question: str, history: Optional[List[dict]] = None) -> dict:
        model = self.pick_model(question)
        msgs = [{"role": "system", "content": ASSISTANT_SYSTEM}] + list(history or [])[-6:] + \
               [{"role": "user", "content": question}]
        used = []
        if self.tool_calling is not False:
            try:
                for _ in range(self.max_steps):
                    out = self.client.chat(model, msgs, purpose="assistant", tools=self.schemas, tool_choice="auto",
                                           max_tokens=500, temperature=0.2, **NO_THINK)
                    msg = out["choices"][0]["message"]
                    calls = msg.get("tool_calls") or []
                    if not calls:
                        self.tool_calling = True if used else self.tool_calling
                        return {"answer": (msg.get("content") or "").strip(), "model": model, "tools": used}
                    msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
                    for c in calls:
                        name = c["function"]["name"]
                        try:
                            args = json.loads(c["function"].get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        result = self._run(name, args)
                        used.append(name)
                        msgs.append({"role": "tool", "tool_call_id": c.get("id", name), "name": name,
                                     "content": json.dumps(result, ensure_ascii=False, default=str)[:6000]})
                return {"answer": "I needed too many steps for that one; try a narrower question.", "model": model,
                        "tools": used}
            except Exception as e:                        # endpoint without tool support: fall back below
                if "tool" not in str(e).lower() and "400" not in str(e):
                    raise
                self.tool_calling = False
        snapshot = {name: self._run(name, {}) for name in self.tools}    # every read-only tool, no arguments
        out = self.client.chat(model, [{"role": "system", "content": ASSISTANT_SYSTEM + "\nShop data:\n" +
                                        json.dumps(snapshot, ensure_ascii=False, default=str)[:12000]},
                                       {"role": "user", "content": question}],
                               purpose="assistant", max_tokens=500, temperature=0.2, **NO_THINK)
        return {"answer": (out["choices"][0]["message"].get("content") or "").strip(), "model": model,
                "tools": list(self.tools), "mode": "snapshot"}


def tool_schemas() -> List[dict]:
    def fn(name, desc, props=None):
        return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
            "type": "object", "properties": props or {}, "required": []}}}
    return [
        fn("get_sessions", "Live camera sessions (one per shopper visit) with totals and lines to check."),
        fn("get_bill", "One session's bill lines with status (ok, verify, disputed, approved, removed).",
           {"session": {"type": "string", "description": "session id; omit for the busiest one"}}),
        fn("get_events", "Recent takes, put-backs, unchecked visits and restock alerts across sessions.",
           {"limit": {"type": "integer"}}),
        fn("get_stock", "Shelf stock from the shop layout: units, minimums, state (ok/low/empty/alert) per shelf.",
           {"zone": {"type": "string", "description": "shelf or camera-zone name; omit for all"}}),
        fn("get_restock_alerts", "Shelves the camera reads as running low, and shelves below their minimum."),
        fn("get_exit_decisions", "Staff decisions at the exit check (keep/remove), with the agent's advice and agreement."),
        fn("get_layout", "The shop layout: shelves, their products and the camera zone that watches each."),
        fn("get_uncertainty", "Per zone: how many takes were uncertain (verify at exit), disputed or unchecked, and why."),
    ]
