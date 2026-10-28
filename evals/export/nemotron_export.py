"""Export the evaluation sets as small labelled crops, so hosted Nemotron can be scored on exactly the
same items as local Qwen2.5-VL-3B (no API key on Kaggle; the API calls run elsewhere).

  shelf/   the 484 shelf-watch crops (Retail Shelf Void Detection, same seed and selection as
           the original Qwen shelf-watch run, evals/data/qwen_shelf_watch.json): empty gap vs stocked shelf
  names/   300 GroceryStoreDataset test photos (seed 7), clean and CCTV-degraded (96 px, JPEG q30),
           each with a shelf of 5 candidate names (true + 4 random); local Qwen's answer on the same items
  visits/  every shelf visit on the 9 labelled Pexels clips (zone mode, hand-drawn zones, labels in evals/data/clip_labels.json):
           the before / after hand crops and local Qwen's P(holding) for each

Output: export.json + the crops (JPEG, long side <= 512)
"""
import glob
import io
import json
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.49", "accelerate", "ultralytics"],
               check=True)
for name, marker in (("vjev", "decider.py"), ("aisle_agent", "pipeline.py")):
    hits = glob.glob(f"/kaggle/input/**/{marker}", recursive=True)
    if not hits:
        for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
            zipfile.ZipFile(z).extractall(f"/kaggle/tmp/src/{os.path.basename(z)[:-4]}")
        hits = glob.glob(f"/kaggle/tmp/src/**/{marker}", recursive=True)
    pkg_dir = os.path.dirname(hits[0])
    if os.path.basename(pkg_dir) != name:
        dst = f"/kaggle/tmp/pkgs/{name}"
        shutil.copytree(pkg_dir, dst, dirs_exist_ok=True)
        pkg_dir = dst
    sys.path.insert(0, os.path.dirname(pkg_dir))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from aisle_agent.logic import Catalogue  # noqa: E402
from aisle_agent.models import VjevEngine  # noqa: E402
from aisle_agent.pipeline import Options, Session  # noqa: E402

OUT, TMP = "/kaggle/working", "/kaggle/tmp/export"
for d in ("shelf", "names", "visits"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
os.makedirs(TMP, exist_ok=True)


def save(img, path, max_side=512):
    im = img.convert("RGB")
    if max(im.size) > max_side:
        k = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))))
    im.save(path, "JPEG", quality=88)


# ------------------------------------------------------------------ shelf watch (same as shelf_watch_check)
def shelf():
    random.seed(0)
    root = "/kaggle/input"
    imgs = [p for p in glob.glob(f"{root}/**/*", recursive=True) if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    pairs = []
    for p in imgs:
        stem = os.path.splitext(p)[0]
        txt, xml = stem.replace("/images/", "/labels/") + ".txt", stem + ".xml"
        boxes = []
        if os.path.exists(txt):
            for line in open(txt):
                v = line.split()
                if len(v) >= 5:
                    cx, cy, w, h = map(float, v[1:5])
                    boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        elif os.path.exists(xml):
            import xml.etree.ElementTree as ET
            r = ET.parse(xml).getroot()
            W, H = float(r.find("size/width").text), float(r.find("size/height").text)
            for o in r.iter("object"):
                b = o.find("bndbox")
                boxes.append([float(b.find("xmin").text) / W, float(b.find("ymin").text) / H,
                              float(b.find("xmax").text) / W, float(b.find("ymax").text) / H])
        else:
            continue
        if boxes:
            pairs.append((p, boxes))

    def iou(a, b):
        ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy + 1e-9)

    def pad(b, f=0.15):
        w, h = b[2] - b[0], b[3] - b[1]
        return [max(0, b[0] - f * w), max(0, b[1] - f * h), min(1, b[2] + f * w), min(1, b[3] + f * h)]

    random.shuffle(pairs)
    rows = []
    for path, voids in pairs[:250]:
        img = Image.open(path).convert("RGB")
        W, H = img.size
        v = max(voids, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        if (v[2] - v[0]) * W < 24 or (v[3] - v[1]) * H < 24:
            continue
        crops = [(1, pad(v))]
        w, h = v[2] - v[0], v[3] - v[1]
        for _ in range(20):
            x, y = random.uniform(0, 1 - w), random.uniform(0, 1 - h)
            cand = [x, y, x + w, y + h]
            if all(iou(cand, o) == 0 for o in voids):
                crops.append((0, pad(cand)))
                break
        for label, b in crops:
            i = len(rows)
            save(img.crop((b[0] * W, b[1] * H, b[2] * W, b[3] * H)), f"{OUT}/shelf/{i}.jpg")
            rows.append({"i": i, "empty": label})
    print("shelf crops", len(rows), flush=True)
    return rows


# ------------------------------------------------------------------ naming by name (GroceryStoreDataset)
def names(eng):
    repo = f"{TMP}/GroceryStoreDataset"
    if not os.path.exists(repo):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/marcusklasson/GroceryStoreDataset.git", repo],
                       check=True)
    d = f"{repo}/dataset"
    rows = [l.strip().split(",") for l in open(f"{d}/classes.csv").read().splitlines()[1:] if l.strip()]
    classes = [r[0].replace("-", " ").replace("_", " ") for r in rows]
    test = []
    for l in open(f"{d}/test.txt").read().splitlines():
        if l.strip():
            p, fine, _ = l.split(",")
            test.append((f"{d}/{p.strip().lstrip('/')}", int(fine)))
    rng = random.Random(7)
    picks = rng.sample(test, 300)
    out = []
    for i, (p, y) in enumerate(picks):
        cands = sorted(rng.sample([c for c in range(len(classes)) if c != y], 4) + [y])
        img = Image.open(p).convert("RGB")
        s = 96 / min(img.size)
        small = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=30)
        cctv = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        save(img, f"{OUT}/names/{i}_clean.jpg")
        cctv.save(f"{OUT}/names/{i}_cctv.jpg", "JPEG", quality=95)          # keep the degradation as is
        opts = [classes[c] for c in cands]
        row = {"i": i, "options": opts, "answer": classes[y]}
        for cond, im in (("clean", img), ("cctv", cctv)):
            t = time.perf_counter()
            a = eng.decider.decide({"assets": {"c": im}, "questions": {"w": {
                "type": "choice", "question": "Which product is this?", "options": opts, "image": "c"}}})["answers"]["w"]
            row[f"qwen_{cond}"] = {"choice": a["choice"], "confidence": a["confidence"],
                                   "ms": round((time.perf_counter() - t) * 1000)}
        out.append(row)
    print("naming items", len(out), flush=True)
    return out


# ------------------------------------------------------------------ shelf visits (same as zone_check)
CLIPS = {
    "29846459": ("https://videos.pexels.com/video-files/29846459/12817260_1280_720_60fps.mp4", [[0.0, 0.0, 0.3, 1.0], [0.65, 0.0, 1.0, 1.0]]),
    "29846460": ("https://videos.pexels.com/video-files/29846460/12817214_1280_720_60fps.mp4", [[0.0, 0.35, 0.55, 1.0]]),
    "29846458": ("https://videos.pexels.com/video-files/29846458/12817190_1280_720_60fps.mp4", [[0.0, 0.0, 0.42, 0.75]]),
    "29814585": ("https://videos.pexels.com/video-files/29814585/12807528_1280_720_60fps.mp4", [[0.28, 0.0, 1.0, 1.0]]),
    "5103996": ("https://videos.pexels.com/video-files/5103996/5103996-hd_1280_720_30fps.mp4", [[0.0, 0.0, 0.27, 1.0]]),
    "39221961": ("https://videos.pexels.com/video-files/39221961/16690543_1280_720_25fps.mp4", [[0.0, 0.1, 1.0, 0.75]]),
    "8465179": ("https://videos.pexels.com/video-files/8465179/8465179-hd_1280_720_25fps.mp4", [[0.5, 0.45, 1.0, 1.0]]),
    "8420772": ("https://videos.pexels.com/video-files/8420772/8420772-hd_1280_720_25fps.mp4", [[0.62, 0.15, 1.0, 0.8]]),
    "8420595": ("https://videos.pexels.com/video-files/8420595/8420595-hd_1280_720_25fps.mp4", [[0.0, 0.0, 0.3, 1.0]]),
}
CATALOGUE = [{"sku": f"p{i}", "name": n, "price": p} for i, (n, p) in enumerate({     # as in zone_check
    "jar of sauce or jam": 189, "boxed pasta or cereal": 145, "carton of milk or juice": 68,
    "vitamin or supplement bottle": 450, "bottle of oil": 240, "shampoo or lotion bottle": 299,
    "packet of biscuits": 40, "canned food": 120, "fresh vegetables": 60, "bag of rice or flour": 330}.items())]


def frames(path, hz=10):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n, nxt = 0, 0.0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if n / fps + 1e-6 >= nxt:
            yield n / fps, f
            nxt += 1.0 / hz
        n += 1
    cap.release()


def visits(eng):
    out = []
    for cid, (url, zboxes) in CLIPS.items():
        path = f"{TMP}/{cid}.mp4"
        if not os.path.exists(path):
            subprocess.run(["wget", "-q", "-U", "Mozilla/5.0", "-O", path, url], check=True)
        zones = [{"id": f"z{i}", "box": b, "skus": []} for i, b in enumerate(zboxes)]
        s = Session(cid, eng, Catalogue(CATALOGUE), Options(async_identify=False), zones=zones)
        for t, f in frames(path):
            _, jobs = s.process(f, t)
            for j in jobs:
                if j.get("kind") != "visit":
                    s.resolve(j)
                    continue
                res = s.resolve(j)
                ev = next((m for m in res if m["type"] in ("take", "put", "touch")), None)
                k = len(out)
                if j["before"] is not None:
                    save(j["before"], f"{OUT}/visits/{k}_before.jpg")
                save(j["after"], f"{OUT}/visits/{k}_after.jpg")
                out.append({"k": k, "clip": cid, "zone": j["zone"], "t": round(j["rise"], 3),
                            "has_before": j["before"] is not None,
                            "qwen": {"kind": ev["type"] if ev else None, "p_before": ev and ev["p_before"],
                                     "p_after": ev and ev["p_after"]}})
        print(cid, "visits so far", len(out), flush=True)
    return out


def main():
    eng = VjevEngine(preset="fast", hold_pixels=224 * 224)
    res = {"shelf": shelf()}
    res["visits"] = visits(eng)
    res["names"] = names(eng)
    json.dump(res, open(f"{OUT}/export.json", "w"), indent=1)
    print({k: len(v) for k, v in res.items()}, flush=True)


main()
