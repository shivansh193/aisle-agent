"""Score hosted NVIDIA Nemotron-3.5-Super-VL (Nebius Token Factory) against local Qwen2.5-VL-3B on the
same exported items (evals/export). Only HTTP calls run here; no model runs locally.

    python evals/eval/nemotron_eval.py --export <dir with export.json> [--only probe|shelf|names|visits]

Key: TOKEN_FACTORY_API_KEY in the environment or .env (never printed). Before each part the cost is
estimated from a few real calls; the run stops if a part is projected to cost more than $5.

Outputs (evals/eval/output/): nemotron_eval.json (metrics, both models), raw_*.jsonl
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from PIL import Image  # noqa: E402
from vjev import Decider  # noqa: E402
from vjev.tokenfactory import VL_MODEL, BackendUnavailable, TokenFactoryBackend, TokenFactoryClient  # noqa: E402

from aisle_agent.logic import STOCK_LEVELS, classify_visit  # noqa: E402
from aisle_agent.models import HAND_QUESTION, STOCK_QUESTION, nemotron_calibration  # noqa: E402

OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
BUDGET = 5.0


def auroc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    return float(sum(abs(conf[m].mean() - correct[m].mean()) * m.mean()
                     for a, b in zip(edges[:-1], edges[1:]) for m in [(conf > a) & (conf <= b)] if m.any()))


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 1) if xs else None


class Runner:
    def __init__(self):
        self.client = TokenFactoryClient(timeout=40, retries=3, max_concurrency=8)
        self.backend = TokenFactoryBackend(VL_MODEL, client=self.client)
        self.decider = Decider(self.backend, calibration=nemotron_calibration())

    def spend(self):
        return self.client.usage["cost_usd"]

    def guard(self, part, done, total, start_cost):
        if done:
            projected = (self.spend() - start_cost) / done * total
            print(f"  {part}: {done}/{total} done, ${self.spend() - start_cost:.4f} so far, projected ${projected:.3f}", flush=True)
            if projected > BUDGET:
                raise SystemExit(f"{part}: projected ${projected:.2f} is over the ${BUDGET} budget; stopping")

    def run(self, part, items, fn, workers=6):
        """fn(item) -> result dict with 'ms'. Cost check after a first batch of 6."""
        start, results, raw = self.spend(), [], open(os.path.join(OUT, f"raw_{part}.jsonl"), "w")
        first, rest = items[:6], items[6:]
        for batch in (first, rest):
            with ThreadPoolExecutor(workers) as ex:
                for r in ex.map(fn, batch):
                    results.append(r)
                    raw.write(json.dumps(r) + "\n")
            self.guard(part, len(results), len(items), start)
        raw.close()
        return results, round(self.spend() - start, 4)


def timed(f):
    t = time.perf_counter()
    try:
        return f(), round((time.perf_counter() - t) * 1000), None
    except BackendUnavailable as e:
        return None, round((time.perf_counter() - t) * 1000), str(e)[:100]


def probe(r):
    """Does the VL endpoint return logprobs for image prompts? (decides logprobs vs votes in vjev)"""
    img = Image.new("RGB", (64, 64), "white")
    out = r.client.chat(VL_MODEL, r.backend._messages(img, "Is this image completely white? Answer Yes or No."),
                        purpose="probe", max_tokens=1, temperature=0, logprobs=True, top_logprobs=5)
    ch = out["choices"][0]
    res = {"model": VL_MODEL, "answer": ch["message"].get("content"), "logprobs_returned": bool(ch.get("logprobs")),
           "top": ch.get("logprobs") and [(t["token"], round(t["logprob"], 3)) for t in
                                          (ch["logprobs"]["content"][0].get("top_logprobs") or [])],
           "usage": out.get("usage")}
    print(json.dumps(res, indent=1))
    return res


def shelf(r, ex, qwen_rows):
    items = ex["shelf"]

    def fn(it):
        img = Image.open(os.path.join(ex["_dir"], "shelf", f"{it['i']}.jpg"))
        d, ms, err = timed(lambda: r.decider.decide({"assets": {"z": img}, "questions": {
            "s": {"type": "score", "question": STOCK_QUESTION, "levels": STOCK_LEVELS, "image": "z"}}}))
        if d is None:
            return {"i": it["i"], "empty": it["empty"], "error": err, "ms": ms}
        dist = d["answers"]["s"]["distribution"]
        level = sum(STOCK_LEVELS.index(k) * p for k, p in dist.items()) / (len(STOCK_LEVELS) - 1)
        return {"i": it["i"], "empty": it["empty"], "level": round(level, 4), "ms": ms, "method": r.backend.last.get("method")}

    res, cost = r.run("shelf", items, fn)
    ok = [x for x in res if "level" in x]
    y = [x["empty"] for x in ok]
    lv = [x["level"] for x in ok]
    out = {"n": len(items), "answered": len(ok), "cost_usd": cost,
           "nemotron": {"auroc": round(auroc([-v for v in lv], y), 4),
                        "acc_at_alert_0.4": round(float(np.mean([(v < 0.4) == bool(e) for v, e in zip(lv, y)])), 4),
                        "ms_p50": pct([x["ms"] for x in ok], 50), "ms_p95": pct([x["ms"] for x in ok], 95)}}
    if qwen_rows and len(qwen_rows) == len(items):
        qy = [q["empty"] for q in qwen_rows]
        if qy == [it["empty"] for it in items]:                      # same crops, same order
            ql = [q["level"] for q in qwen_rows]
            out["qwen_local"] = {"auroc": round(auroc([-v for v in ql], qy), 4),
                                 "acc_at_alert_0.4": round(float(np.mean([(v < 0.4) == bool(e) for v, e in zip(ql, qy)])), 4),
                                 "ms_mean_T4": 331.4}
        else:
            out["qwen_local"] = "labels differ from the original run; not comparable"
    return out


def names(r, ex):
    items = ex["names"]

    def fn(it):
        row = {"i": it["i"]}
        for cond in ("clean", "cctv"):
            img = Image.open(os.path.join(ex["_dir"], "names", f"{it['i']}_{cond}.jpg"))
            d, ms, err = timed(lambda: r.decider.decide({"assets": {"c": img}, "questions": {"w": {
                "type": "choice", "question": "Which product is this?", "options": it["options"], "image": "c"}}}))
            row[cond] = ({"error": err, "ms": ms} if d is None else
                         {"choice": d["answers"]["w"]["choice"], "confidence": d["answers"]["w"]["confidence"], "ms": ms,
                          "probs": [d["answers"]["w"]["probabilities"][o] for o in it["options"]]})
        row["ms"] = row["clean"]["ms"]
        return row

    res, cost = r.run("names", items, fn)
    by = {x["i"]: x for x in res}
    out = {"n": len(items), "cost_usd": cost, "candidates": 5}
    for cond in ("clean", "cctv"):
        ok = [(it, by[it["i"]][cond]) for it in items if "choice" in by[it["i"]][cond]]
        corr = [a["choice"] == it["answer"] for it, a in ok]
        out[f"nemotron_{cond}"] = {"top1": round(float(np.mean(corr)), 4), "ece": round(ece([a["confidence"] for _, a in ok], corr), 4),
                                   "answered": len(ok), "ms_p50": pct([a["ms"] for _, a in ok], 50),
                                   "ms_p95": pct([a["ms"] for _, a in ok], 95)}
        q = [(it[f"qwen_{cond}"]["choice"] == it["answer"], it[f"qwen_{cond}"]["confidence"]) for it in items]
        out[f"qwen_local_{cond}"] = {"top1": round(float(np.mean([c for c, _ in q])), 4),
                                     "ece": round(ece([p for _, p in q], [c for c, _ in q]), 4),
                                     "ms_mean_T4": round(float(np.mean([it[f"qwen_{cond}"]["ms"] for it in items])), 1)}
    out["calibration"] = calibrate(items, by)
    return out


def calibrate(items, by):
    """One choice temperature for Nemotron, fitted on even-numbered items (clean + CCTV), scored on odd ones."""
    from vjev import fit_temperature
    from vjev.calibration import softmax
    rows = {"fit": [], "test": []}
    for it in items:
        for cond in ("clean", "cctv"):
            a = by[it["i"]][cond]
            if "probs" in a:
                z = np.log(np.clip(np.array(a["probs"], float), 1e-9, 1))
                rows["fit" if it["i"] % 2 == 0 else "test"].append((z, it["options"].index(it["answer"]), cond))
    T = fit_temperature([z for z, _, _ in rows["fit"]], [y for _, y, _ in rows["fit"]])
    res = {"temperature": round(T, 4), "fitted_on": len(rows["fit"]), "scored_on": len(rows["test"])}
    for cond in ("clean", "cctv"):
        test = [(z, y) for z, y, c in rows["test"] if c == cond]
        for tag, t in (("before", 1.0), ("after", T)):
            P = [softmax(z, t) for z, _ in test]
            res[f"{cond}_ece_{tag}"] = round(ece([p.max() for p in P], [p.argmax() == y for p, (_, y) in zip(P, test)]), 4)
    json.dump({"choice_temperature": T, "score_temperature": 1.0, "noul_platt": {"mu": [0.0], "sd": [1.0], "w": [1.0, 0.0]},
               "_note": "choice temperature fitted on held-out GroceryStoreDataset naming items (evals/eval)"},
              open(os.path.join(ROOT, "aisle_agent", "nemotron_calibration.json"), "w"), indent=1)
    return res


def visits(r, ex, labels):
    items = ex["visits"]
    cal = r.decider.cal

    def fn(it):
        imgs = ([Image.open(os.path.join(ex["_dir"], "visits", f"{it['k']}_before.jpg"))] if it["has_before"] else []) + \
               [Image.open(os.path.join(ex["_dir"], "visits", f"{it['k']}_after.jpg"))]
        q = HAND_QUESTION + " Answer Yes or No."
        zs, ms, err = timed(lambda: r.backend.batch_next_token_logits([(im, q, ["Yes", "No"]) for im in imgs]))
        if zs is None:
            return {**it, "nemotron": {"kind": "unchecked", "error": err}, "ms": ms}
        ps = [float(cal.noul_platt([[float(z[0] - z[1])]])[0]) for z in zs]
        pb, pa = (ps[0], ps[1]) if it["has_before"] else (None, ps[0])
        return {**it, "nemotron": {"kind": classify_visit(pb, pa, 0.6, 0.5), "p_before": pb and round(pb, 4),
                                   "p_after": round(pa, 4)}, "ms": ms}

    res, cost = r.run("visits", items, fn)

    def score(kind_of):
        tp = fp = fn = 0
        for cid, L in labels.items():
            takes = [x["t"] for x in res if x["clip"] == cid and kind_of(x) == "take"]
            used = set()
            for g in L["takes"]:
                m = [e for e in takes if abs(e - g) <= 1.0 and e not in used]
                if m:
                    used.add(m[0]); tp += 1
                else:
                    fn += 1
            fp += len([e for e in takes if e not in used and not any(abs(e - i) <= 1.0 for i in L["ignore"])])
        p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return {"takes_found": tp, "false_takes": fp, "missed": fn, "precision": round(p, 3), "recall": round(rc, 3)}

    kinds = lambda key: {k: sum(1 for x in res if (x[key] or {}).get("kind") == k) for k in ("take", "put", "touch", "unchecked")}
    return {"visits": len(items), "cost_usd": cost, "labelled_takes": sum(len(L["takes"]) for L in labels.values()),
            "nemotron": {**score(lambda x: x["nemotron"]["kind"]), "events": kinds("nemotron"),
                         "ms_p50_per_visit": pct([x["ms"] for x in res], 50), "ms_p95_per_visit": pct([x["ms"] for x in res], 95)},
            "qwen_local": {**score(lambda x: x["qwen"]["kind"]), "events": kinds("qwen")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=os.path.join(ROOT, "evals", "export", "output"))
    ap.add_argument("--only", default="probe,shelf,names,visits")
    a = ap.parse_args()
    parts = a.only.split(",")
    r = Runner()
    path = os.path.join(OUT, "nemotron_eval.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    if "probe" in parts:
        res["probe"] = probe(r)
    ex = None
    if set(parts) & {"shelf", "names", "visits"}:
        ex = json.load(open(os.path.join(a.export, "export.json")))
        ex["_dir"] = a.export
    if "shelf" in parts:
        q = json.load(open(os.path.join(ROOT, "evals", "data", "qwen_shelf_watch.json")))["rows"]
        res["shelf_watch"] = shelf(r, ex, q)
        print(json.dumps(res["shelf_watch"], indent=1), flush=True)
    if "names" in parts:
        res["naming_by_name"] = names(r, ex)
        print(json.dumps(res["naming_by_name"], indent=1), flush=True)
    if "visits" in parts:
        lab = {k: v for k, v in json.load(open(os.path.join(ROOT, "evals", "data", "clip_labels.json"))).items()
               if not k.startswith("_")}
        res["zone_takes"] = visits(r, ex, lab)
        print(json.dumps(res["zone_takes"], indent=1), flush=True)
    res["usage_total"] = r.client.usage
    res["logprobs_supported"] = r.backend.logprobs_supported
    json.dump(res, open(path, "w"), indent=1)
    print("saved", path, "spent this run: $%.4f" % r.spend())


if __name__ == "__main__":
    main()
