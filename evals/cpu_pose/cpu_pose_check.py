"""Can Aisle Agent run without a GPU once the vision-language model is hosted (Nemotron on Token Factory)?
On Kaggle's CPU (4 vCPU): YOLO11n-pose + ByteTrack fps at 320 / 416 / 640 px on a real shelf clip, and
SigLIP so400m embedding time per take (the only other local model). Output: cpu_pose.json
"""
import json
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "lap"], check=True)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT, TMP = "/kaggle/working", "/kaggle/tmp"
os.makedirs(TMP, exist_ok=True)
clip = f"{TMP}/clip.mp4"
subprocess.run(["wget", "-q", "-U", "Mozilla/5.0", "-O", clip,
                "https://videos.pexels.com/video-files/8420772/8420772-hd_1280_720_25fps.mp4"], check=True)
cap, frames = cv2.VideoCapture(clip), []
while len(frames) < 150:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
res = {"cpu": os.cpu_count(), "torch_threads": torch.get_num_threads(), "frames": len(frames), "pose": {}}
from ultralytics import YOLO  # noqa: E402
for imgsz in (320, 416, 640):
    m = YOLO("yolo11n-pose.pt")
    for f in frames[:5]:
        m.track(f, persist=True, tracker="bytetrack.yaml", imgsz=imgsz, classes=[0], verbose=False, device="cpu")
    ts = []
    for f in frames[5:]:
        t = time.perf_counter()
        r = m.track(f, persist=True, tracker="bytetrack.yaml", imgsz=imgsz, classes=[0], verbose=False, device="cpu")[0]
        ts.append((time.perf_counter() - t) * 1000)
    res["pose"][imgsz] = {"ms_p50": round(float(np.percentile(ts, 50)), 1), "ms_p95": round(float(np.percentile(ts, 95)), 1),
                          "fps": round(1000 / float(np.mean(ts)), 1), "people_last": int(len(r.boxes) if r.boxes is not None else 0)}
    print(imgsz, res["pose"][imgsz], flush=True)
from transformers import AutoModel, AutoProcessor  # noqa: E402
from PIL import Image  # noqa: E402
name = "google/siglip-so400m-patch14-384"
proc, model = AutoProcessor.from_pretrained(name), AutoModel.from_pretrained(name).eval()
img = Image.fromarray(frames[0][:, :, ::-1]).crop((400, 200, 624, 424))
ts = []
for i in range(6):
    t = time.perf_counter()
    with torch.no_grad():
        model.get_image_features(**proc(images=[img], return_tensors="pt"))
    if i:
        ts.append((time.perf_counter() - t) * 1000)
res["siglip_ms_per_take"] = round(float(np.mean(ts)), 1)
print(json.dumps(res, indent=1))
json.dump(res, open(f"{OUT}/cpu_pose.json", "w"), indent=1)
