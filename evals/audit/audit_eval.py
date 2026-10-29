"""Shelf audit evaluation: price labels and wrong-shelf products, scored through the app's own code
(NemotronEngine.read_labels / check_misplaced, logic.match_label, the pipeline's ₹1 mismatch rule).

    python evals/audit/audit_eval.py --shelves <dir of shelf photos> --frames <dir of freezer frames> \
        --products <dir with export.json and names/>   [--only price|misplaced] [--build-only]

There is no public set of shelf photos with known label prices, so both sets are built here, from a fixed seed:

price      Real shelf photos (Retail Shelf Void Detection, and frames of the demo's freezer footage) with two
           printed shelf labels pasted along a shelf edge: product name and a ₹ price. Five sizes, given as the
           height in pixels of the price digits in the image sent to the model, each clean and CCTV-degraded.
           30% of labels show a price that differs from the catalogue; those are the mismatches to catch.
misplaced  Crops of the freezer footage. The shelf is a yoghurt chiller. Three cases:
           - clean crop, list = what is really there (should say No: false alarms);
           - the same crop with a non-dairy product photo pasted on (should say Yes);
           - clean crop, list = the demo layout's freezer list (frozen food, trays, milk), which is wrong for
             this shelf, so every yoghurt is a real misplaced product (should say Yes).
           The pasted products are cut-out rectangles, so they are easier to spot than a real misplaced item.

Key: TOKEN_FACTORY_API_KEY in the environment or .env (never printed). Stops if projected cost > $5.
Outputs (evals/audit/output/): audit_eval.json, raw_*.jsonl; the built images go to --work (not committed).
"""
import argparse
import glob
import io
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

OUT = os.path.join(HERE, "output")
BUDGET = 5.0
DIGIT_PX = [6, 9, 12, 16, 24]                 # price digit height in the image the model sees
WIDTH = 1024                                  # the audit sends crops at up to 1024 px
CHILLER = ["cups and tubs of yoghurt", "yoghurt multipacks", "dairy desserts"]
DEMO_FREEZER = ["frozen food pack", "tray of sauce or ready meal", "carton of milk or juice"]
DAIRY = ("yog", "yoghurt", "milk", "juice", "oat", "cream", "soy", "kefir", "fil", "smoothie", "dairy", "ghurt")
FREEZER_ZONE = (0.01, 0.36, 0.54, 0.99)       # the zone drawn in the demo (frame fractions)


def font(px, bold=False):
    for f in (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(f, max(4, int(px)))
        except OSError:
            continue
    return ImageFont.load_default()


def label_card(name, price, digit_px, rng):
    """A printed shelf label: name above, '₹ price' large below. Size follows the price digit height."""
    h = digit_px / 0.42
    w = int(round(h * 2.6))
    h = int(round(h))
    yellow = rng.random() < 0.6
    card = Image.new("RGB", (w, h), (255, 216, 0) if yellow else (250, 250, 246))
    d = ImageDraw.Draw(card)
    nf, pf = font(h * 0.17), font(digit_px / 0.72, bold=True)     # cap height ≈ 0.72 × font size
    words, line, lines = name.split(), "", []
    for wd in words:
        if d.textlength((line + " " + wd).strip(), font=nf) > w * 0.92 and line:
            lines.append(line)
            line = wd
        else:
            line = (line + " " + wd).strip()
    lines.append(line)
    for i, ln in enumerate(lines[:2]):
        d.text((w * 0.04, h * (0.05 + 0.19 * i)), ln, font=nf, fill=(20, 20, 20))
    d.text((w * 0.04, h * 0.45), f"₹{price}", font=pf, fill=(20, 20, 20) if yellow else (190, 20, 20))
    d.rectangle([0, 0, w - 1, h - 1], outline=(120, 120, 120))
    return card.rotate(rng.uniform(-3, 3), expand=True, fillcolor=(0, 0, 0)), w, h


def degrade(img, rng):
    """CCTV-like: lower-resolution sensor, soft focus, noise, heavy compression."""
    w, h = img.size
    im = img.resize((int(w * 0.6), int(h * 0.6)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    im = im.filter(ImageFilter.GaussianBlur(0.8))
    a = np.asarray(im).astype(np.float32) + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 6, (h, w, 3))
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=35)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def to_width(img, w=WIDTH):
    return img.convert("RGB").resize((w, round(img.height * w / img.width)), Image.LANCZOS)


def build_price(shelves, frames, names, work, rng):
    bgs = sorted(glob.glob(os.path.join(shelves, "*.jpg")))[:30] + sorted(glob.glob(os.path.join(frames, "*.jpg")))[::3][:10]
    items = []
    for bi, bg in enumerate(bgs):
        base = Image.open(bg)
        if os.path.dirname(bg) == os.path.normpath(frames) or bg.startswith(frames):     # the shelf zone, as the app crops it
            W, H = base.size
            base = base.crop((int(FREEZER_ZONE[0] * W), int(FREEZER_ZONE[1] * H), int(FREEZER_ZONE[2] * W), int(FREEZER_ZONE[3] * H)))
        base = to_width(base)
        for dpx in DIGIT_PX:
            prods = rng.sample(names, 3)
            cat = {n: rng.choice([25, 35, 40, 45, 60, 68, 85, 99, 120, 145, 180, 189, 220, 249, 315, 450]) for n in prods}
            img, labels = base.copy(), []
            y = int(img.height * rng.uniform(0.55, 0.85))
            xs = [rng.uniform(0.05, 0.4), rng.uniform(0.55, 0.85)]
            for n, xf in zip(prods[:2], xs):
                wrong = rng.random() < 0.3
                shown = cat[n] if not wrong else rng.choice([p for p in (cat[n] - 10, cat[n] + 5, cat[n] + 20, cat[n] - 5) if p > 0])
                card, w, h = label_card(n, shown, dpx, rng)
                mask = card.convert("L").point(lambda v: 255 if v > 0 else 0)
                img.paste(card, (int(img.width * xf), y + rng.randint(-4, 4)), mask)
                labels.append({"name": n, "shown": shown, "catalogue": cat[n], "mismatch": wrong})
            for q in ("clean", "cctv"):
                out = img if q == "clean" else degrade(img, rng)
                path = os.path.join(work, f"price_{bi:02d}_{dpx:02d}_{q}.jpg")
                out.save(path, quality=90)
                items.append({"path": path, "bg": os.path.basename(bg), "digit_px": dpx, "quality": q,
                              "products": [{"name": n, "price": cat[n]} for n in prods], "labels": labels})
    return items


def build_misplaced(frames, products, work, rng):
    ex = json.load(open(os.path.join(products, "export.json")))
    foreign = [(it["i"], it["answer"]) for it in ex["names"] if not any(k in it["answer"].lower() for k in DAIRY)]
    items = []
    for fi, f in enumerate(sorted(glob.glob(os.path.join(frames, "*.jpg")))):
        full = Image.open(f).convert("RGB")
        W, H = full.size
        for ci in range(2):
            j = [rng.uniform(-0.03, 0.03) for _ in range(4)] if ci else [0, 0, 0, 0]
            box = [min(1, max(0, a + b)) for a, b in zip(FREEZER_ZONE, j)]
            crop = full.crop((int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)))
            stem = f"mis_{fi:02d}_{ci}"
            clean = os.path.join(work, stem + "_clean.jpg")
            crop.save(clean, quality=90)
            i, name = rng.choice(foreign)
            prod = Image.open(os.path.join(products, "names", f"{i}_clean.jpg")).convert("RGB")
            s = prod.width
            prod = prod.crop((s * 0.12, s * 0.12, s * 0.88, s * 0.88))
            pw = int(crop.width * rng.uniform(0.16, 0.26))
            prod = prod.resize((pw, pw), Image.LANCZOS)
            pasted = crop.copy()
            pasted.paste(prod, (rng.randint(0, crop.width - pw), rng.randint(int(crop.height * 0.3), crop.height - pw)))
            pos = os.path.join(work, stem + "_pasted.jpg")
            pasted.save(pos, quality=90)
            items += [{"path": clean, "case": "clean", "expected": CHILLER, "truth": 0},
                      {"path": pos, "case": "pasted", "expected": CHILLER, "truth": 1, "pasted": name},
                      {"path": clean, "case": "layout_wrong", "expected": DEMO_FREEZER, "truth": 1}]
    return items


class Runner:
    def __init__(self):
        from vjev import Decider
        from vjev.tokenfactory import VL_MODEL, TokenFactoryBackend, TokenFactoryClient
        from aisle_agent.models import NemotronEngine, nemotron_calibration
        self.client = TokenFactoryClient(timeout=60, retries=3, max_concurrency=8)
        eng = NemotronEngine.__new__(NemotronEngine)          # only the audit methods: no pose or photo models
        eng.backend = TokenFactoryBackend(VL_MODEL, client=self.client)
        eng.decider = Decider(eng.backend, calibration=nemotron_calibration())
        self.engine = eng

    def spend(self):
        return self.client.usage["cost_usd"]

    def run(self, part, items, fn, workers=6):
        start, results = self.spend(), []
        raw = open(os.path.join(OUT, f"raw_{part}.jsonl"), "w")
        for batch in (items[:6], items[6:]):
            with ThreadPoolExecutor(workers) as ex:
                for r in ex.map(fn, batch):
                    results.append(r)
                    raw.write(json.dumps(r) + "\n")
            done = len(results)
            projected = (self.spend() - start) / done * len(items)
            print(f"  {part}: {done}/{len(items)}, ${self.spend() - start:.4f}, projected ${projected:.3f}", flush=True)
            if projected > BUDGET:
                raise SystemExit(f"{part}: projected ${projected:.2f} is over the ${BUDGET} budget; stopping")
        raw.close()
        return results, round(self.spend() - start, 4)


def score_price(results):
    from aisle_agent.logic import match_label

    class P:                                                   # what match_label needs from a product
        def __init__(self, name, price):
            self.name, self.sku, self.price = name, name, price

    cells = {}
    for r in results:
        if r.get("read") is None:
            continue
        prods = [P(p["name"], p["price"]) for p in r["products"]]
        flagged = {}                                           # product name -> label price, as the pipeline would flag
        read_ok = 0
        prices_read = [float(lab["price"]) for lab in r["read"]]
        for lab in r["read"]:
            prod = match_label(lab.get("text", ""), prods)
            if prod is None:
                continue
            truth = next((t for t in r["labels"] if t["name"] == prod.name), None)
            if truth and abs(lab["price"] - truth["shown"]) < 1:
                read_ok += 1
            if abs(float(lab["price"]) - float(prod.price)) >= 1:
                flagged[prod.name] = lab["price"]
        c = cells.setdefault((r["digit_px"], r["quality"]), {"labels": 0, "read_right": 0, "price_read": 0, "mismatches": 0, "caught": 0,
                                                             "ok_labels": 0, "false_flags": 0, "images": 0, "none_read": 0})
        c["images"] += 1
        c["none_read"] += not r["read"]
        c["labels"] += len(r["labels"])
        c["read_right"] += min(read_ok, len(r["labels"]))
        c["price_read"] += sum(any(abs(x - t["shown"]) < 1 for x in prices_read) for t in r["labels"])
        for t in r["labels"]:
            if t["mismatch"]:
                c["mismatches"] += 1
                c["caught"] += t["name"] in flagged
            else:
                c["ok_labels"] += 1
                c["false_flags"] += t["name"] in flagged
        c["false_flags"] += sum(1 for n in flagged if n not in {t["name"] for t in r["labels"]})   # flags on unlabelled products
    rows = []
    for (dpx, q), c in sorted(cells.items()):
        rows.append({"digit_px": dpx, "quality": q, "images": c["images"],
                     "prices_read": round(c["price_read"] / c["labels"], 3),
                     "labels_read_right": round(c["read_right"] / c["labels"], 3),
                     "mismatches_caught": f"{c['caught']}/{c['mismatches']}",
                     "false_flags": f"{c['false_flags']}/{c['ok_labels']}",
                     "images_nothing_read": round(c["none_read"] / c["images"], 3)})
    return rows


def auroc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels)
    pos, neg = s[y == 1], s[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


ALIASES = {"conference": "pear", "granny smith": "apple", "pink lady": "apple", "royal gala": "apple",
           "satsumas": ("satsuma", "mandarin", "clementine", "tangerine", "orange"), "aubergine": ("aubergine", "eggplant"),
           "brown cap mushroom": "mushroom", "honeydew melon": "melon", "red grapefruit": "grapefruit",
           "solid potato": "potato", "sweet potato": ("sweet potato", "yam"), "vine tomato": "tomato",
           "green bell pepper": ("pepper", "capsicum"), "red bell pepper": ("pepper", "capsicum"),
           "yellow bell pepper": ("pepper", "capsicum"), "yellow onion": "onion", "passion fruit": "passion"}


def named_right(case, pasted, item):
    """Lenient: the reply names the pasted product's kind (apple for Pink Lady), or yoghurt for the wrong layout."""
    if not item:
        return False
    item = item.lower()
    if case == "layout_wrong":
        return any(w in item for w in ("yog", "yoghurt", "dessert", "dairy"))
    want = ALIASES.get(pasted.lower(), pasted.lower())
    want = (want,) if isinstance(want, str) else want
    return any(w.rstrip("s") in item for w in want)


def score_misplaced(results, at=0.6):
    by = {}
    for r in results:
        by.setdefault(r["case"], []).append(r)
    out = {c: {"n": len(rs), "flagged_at_0.6": round(float(np.mean([r["p"] >= at for r in rs])), 3),
               "p_median": round(float(np.median([r["p"] for r in rs])), 3)} for c, rs in by.items()}
    for c in ("pasted", "layout_wrong"):
        flagged = [r for r in by.get(c, []) if r["p"] >= 0.5]
        if flagged:
            out[c]["named_right"] = f"{sum(named_right(c, r.get('pasted') or '', r['item']) for r in flagged)}/{len(flagged)}"
    out.get("clean", {})["flagged_at_0.5"] = round(float(np.mean([r["p"] >= 0.5 for r in by.get("clean", [])])), 3)
    neg = [r["p"] for r in by.get("clean", [])]
    for c in ("pasted", "layout_wrong"):
        if c in by and neg:
            out[c]["auroc_vs_clean"] = round(auroc([r["p"] for r in by[c]] + neg, [1] * len(by[c]) + [0] * len(neg)), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shelves", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--products", required=True)
    ap.add_argument("--work", default=os.path.join(HERE, "work"))
    ap.add_argument("--only", choices=["price", "misplaced"])
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--no-list", action="store_true", help="read labels without the shelf's product list (the old prompt)")
    ap.add_argument("--rescore", action="store_true", help="score the saved raw_*.jsonl again, no model calls")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    rng = random.Random(7)
    names = sorted({it["answer"] for it in json.load(open(os.path.join(a.products, "export.json")))["names"]})
    price = build_price(a.shelves, a.frames, names, a.work, rng)
    mis = build_misplaced(a.frames, a.products, a.work, rng)
    print(f"built {len(price)} price images, {len(mis)} misplaced items in {a.work}")
    if a.build_only:
        return
    path = os.path.join(OUT, "audit_eval.json")
    report = json.load(open(path)) if os.path.exists(path) else {}
    if a.rescore:
        for part, score, key in (("price", score_price, "rows"), ("misplaced", score_misplaced, "cases")):
            raw = os.path.join(OUT, f"raw_{part}.jsonl")
            if os.path.exists(raw) and part in report:
                report[part][key] = score([json.loads(x) for x in open(raw)])
        json.dump(report, open(path, "w"), indent=1, ensure_ascii=False)
        print(json.dumps(report, indent=1, ensure_ascii=False))
        return
    r = Runner()

    if a.only in (None, "price"):
        def fn(it):
            t = time.time()
            try:
                read = r.engine.read_labels(Image.open(it["path"]), None if a.no_list else [x["name"] for x in it["products"]])
            except Exception as e:                             # one failed call doesn't end the run; counted below
                return {**it, "read": None, "error": str(e)[:200], "ms": int((time.time() - t) * 1000)}
            return {**it, "read": read, "ms": int((time.time() - t) * 1000)}
        res, cost = r.run("price", price, fn)
        report["price"] = {"rows": score_price(res), "cost_usd": cost, "images": len(res),
                           "failed_calls": sum(x.get("read") is None for x in res),
                           "ms_p50": int(np.median([x["ms"] for x in res]))}
    if a.only in (None, "misplaced"):
        def fn(it):
            t = time.time()
            m = r.engine.check_misplaced(Image.open(it["path"]), it["expected"])
            return {**it, "p": m["p"], "item": m["item"], "ms": int((time.time() - t) * 1000)}
        res, cost = r.run("misplaced", mis, fn)
        report["misplaced"] = {"cases": score_misplaced(res), "cost_usd": cost, "items": len(res),
                               "ms_p50": int(np.median([x["ms"] for x in res])),
                               "named_examples": [[x.get("pasted") or x["case"], x["item"]] for x in res if x["p"] >= 0.5][:16]}
    json.dump(report, open(path, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
