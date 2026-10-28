"""Local Qwen2.5-VL-3B (T4) shelf-watch reading on the exact 484 crops Nemotron was scored on
(evals/export -> dataset aisle-agent-shelf-crops), for a like-for-like comparison. Output: qwen_shelf.json
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.49", "accelerate", "ultralytics"], check=True)
for name, marker in (("vjev", "decider.py"), ("shelfsense_live", "pipeline.py")):
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
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from shelfsense_live.logic import STOCK_LEVELS  # noqa: E402
from shelfsense_live.models import VjevEngine  # noqa: E402

items = json.load(open(glob.glob("/kaggle/input/**/labels.json", recursive=True)[0]))
hits = glob.glob("/kaggle/input/**/shelf/0.jpg", recursive=True)
if not hits:                                     # uploaded zipped: unpack it
    for z in glob.glob("/kaggle/input/**/shelf*.zip", recursive=True):
        zipfile.ZipFile(z).extractall("/kaggle/tmp/crops")
    hits = glob.glob("/kaggle/tmp/crops/**/0.jpg", recursive=True)
root = os.path.dirname(os.path.dirname(hits[0]))
eng = VjevEngine(preset="fast")
rows = []
for it in items:
    img = Image.open(f"{root}/shelf/{it['i']}.jpg").convert("RGB")
    t = time.perf_counter()
    dist = eng.stock([img])[0]
    ms = (time.perf_counter() - t) * 1000
    level = sum(STOCK_LEVELS.index(k) * p for k, p in dist.items()) / (len(STOCK_LEVELS) - 1)
    rows.append({"i": it["i"], "empty": it["empty"], "level": round(level, 4), "ms": round(ms, 1)})
y, lv = np.array([r["empty"] for r in rows]), np.array([r["level"] for r in rows])
pos, neg = -lv[y == 1], -lv[y == 0]
auroc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
out = {"n": len(rows), "auroc": round(auroc, 4), "acc_at_alert_0.4": round(float(np.mean((lv < 0.4) == (y == 1))), 4),
       "ms_p50": round(float(np.percentile([r["ms"] for r in rows[5:]], 50)), 1), "rows": rows}
json.dump(out, open("/kaggle/working/qwen_shelf.json", "w"))
print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
