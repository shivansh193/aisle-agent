"""Perception: who is where (every frame) and, per shopper, holding / what (on demand).

An engine owns the shared GPU models and hands each session its own tracker (track ids are
per stream). Engines implement:

    new_tracker() -> callable(frame) -> [Track]
    p_hold(frame, tracks) -> [float]                       P(holding a product), one per track (no zones)
    p_hand(crops) -> [float]                               P(this hand holds a product), one per hand crop
    identify(frame, track, catalogue, region=None) -> {sku, confidence, box (frame fractions) | None, box_confidence}
                                                           region: pixel box of the hand, when known

`VjevEngine` is the real one; `StubEngine` is scripted, for tests and for running the API
without a GPU.
"""
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .logic import STOCK_LEVELS, Catalogue, rank_products

HOLD_QUESTION = "Is this person holding a product in their hand?"
HAND_QUESTION = "Is this hand holding a product?"
PRODUCT_TARGET = "the product in the person's hand"
PRODUCT_QUESTION = "Which product is the person holding?"
STOCK_QUESTION = "How well stocked with products is this shelf area?"
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10


@dataclass
class Track:
    id: int
    box: Tuple[float, float, float, float]          # pixels, x0 y0 x1 y1
    conf: float
    keypoints: Optional[np.ndarray] = None          # (17, 3): x, y, confidence (COCO order)


def clamp_box(box, w, h):
    x0, y0, x1, y1 = box
    return (int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1)))


MIN_CROP = 56


def crop(frame: np.ndarray, box):
    """RGB PIL crop of a pixel box from a BGR frame."""
    from PIL import Image
    w, h = frame.shape[1], frame.shape[0]
    x0, y0, x1, y1 = clamp_box(box, w, h)
    x0, y0 = min(x0, w - 2), min(y0, h - 2)                      # a hand past the frame edge still gets a sliver
    x1, y1 = max(x1, x0 + 2), max(y1, y0 + 2)
    im = Image.fromarray(np.ascontiguousarray(frame[y0:y1, x0:x1, ::-1]))
    if min(im.size) < MIN_CROP:            # the VLM's vision patches need >= 28 px a side
        k = MIN_CROP / min(im.size)
        im = im.resize((max(MIN_CROP, round(im.width * k)), max(MIN_CROP, round(im.height * k))), Image.BICUBIC)
    return im


def person_region(box, pad: float = 0.15):
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return (x0 - px, y0 - py, x1 + px, y1 + py)


def crop_person(frame: np.ndarray, box, pad: float = 0.15):
    """RGB PIL crop around a person box, plus its (x0, y0, w, h) in the frame."""
    x0, y0, x1, y1 = clamp_box(person_region(box, pad), frame.shape[1], frame.shape[0])
    return crop(frame, (x0, y0, x1, y1)), (x0, y0, x1 - x0, y1 - y0)


def hand_regions(track: Track, min_conf: float = 0.3) -> Dict[int, Tuple[float, float, float, float]]:
    """{0: left, 1: right} square pixel boxes around each visible hand: centred a little past
    the wrist along the forearm (where a held product is), sized by forearm length and person height."""
    if track.keypoints is None:
        return {}
    ph = track.box[3] - track.box[1]
    out = {}
    for k, (e, w) in enumerate(((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST))):
        wx, wy, wc = track.keypoints[w]
        if wc < min_conf:
            continue
        ex, ey, ec = track.keypoints[e]
        if ec >= min_conf:
            fx, fy = wx - ex, wy - ey
            arm = float(np.hypot(fx, fy))
            cx, cy = wx + 0.35 * fx, wy + 0.35 * fy
            side = float(np.clip(1.3 * arm, 0.18 * ph, 0.45 * ph))
        else:
            cx, cy, side = wx, wy, 0.3 * ph
        out[k] = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    return out


def decode_image(src: str, max_bytes: int = 5_000_000):
    """Catalogue photo -> RGB PIL image. Accepts data URIs and base64. http(s) URLs are fetched
    only when AISLE_FETCH_URLS=1 (off by default: the server would fetch arbitrary URLs)."""
    import base64
    import io
    import os
    from PIL import Image
    if src.startswith(("http://", "https://")):
        if os.environ.get("AISLE_FETCH_URLS") != "1":
            raise ValueError("image URLs are disabled; send the photo as a data URI")
        import urllib.request
        with urllib.request.urlopen(src, timeout=10) as r:
            data = r.read(max_bytes + 1)
    else:
        data = base64.b64decode(src.split(",", 1)[1] if src.startswith("data:") else src)
    if len(data) > max_bytes:
        raise ValueError("image too large")
    return Image.open(io.BytesIO(data)).convert("RGB")


class Embedder:
    """Image embeddings for matching a held product against catalogue photos.

    Temperatures were fitted on GroceryStoreDataset (an earlier naming study): with one catalogue
    photo per product, SigLIP so400m names the product 98.5% of the time among a shelf of 5
    on clean crops; DINOv2-base holds up better on CCTV-quality crops (87.0% with 3 photos)."""

    MODELS = {"siglip": ("google/siglip-so400m-patch14-384", 0.0192), "dino": ("facebook/dinov2-base", 0.033)}

    def __init__(self, kind: str = "siglip", device: str = "cuda"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        name, self.temperature = self.MODELS[kind]
        self.kind, self.torch, self.device = kind, torch, device
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()

    def __call__(self, images) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            x = self.proc(images=list(images), return_tensors="pt")["pixel_values"].to(self.device, torch.float16)
            if self.kind == "siglip":
                e = self.model.get_image_features(pixel_values=x)
                e = e if torch.is_tensor(e) else e.pooler_output
            else:
                e = self.model(pixel_values=x).last_hidden_state[:, 0]
            return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()


class PoseTracker:
    """YOLO11n-pose + ByteTrack. One instance per stream (the tracker keeps state)."""

    def __init__(self, weights: str = "yolo11n-pose.pt", imgsz: int = 640, conf: float = 0.3):
        from ultralytics import YOLO
        self.model, self.imgsz, self.conf = YOLO(weights), imgsz, conf

    def __call__(self, frame: np.ndarray) -> List[Track]:
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=self.imgsz, conf=self.conf,
                             classes=[0], verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            return []
        out = []
        kp = r.keypoints
        for k in range(len(r.boxes)):
            pts = None
            if kp is not None:
                xy = kp.xy[k].cpu().numpy()
                cf = kp.conf[k].cpu().numpy() if kp.conf is not None else np.ones(len(xy))
                pts = np.concatenate([xy, cf[:, None]], 1)
            out.append(Track(int(r.boxes.id[k]), tuple(float(v) for v in r.boxes.xyxy[k].tolist()),
                             float(r.boxes.conf[k]), pts))
        return out


class VjevEngine:
    """Holding: one batched yes/no pass over small crops (each visible hand, or the whole
    person when no hand is visible); a person's P(hold) is the max over their crops.
    Product: vjev box + choice over the catalogue names (step 1 identifier)."""

    def __init__(self, preset: str = "fast", pose_weights: str = "yolo11n-pose.pt", hold_pixels: int = 224 * 224,
                 hold_crop: str = "hands", model: str = "Qwen/Qwen2.5-VL-3B-Instruct", embedder: str = "siglip"):
        from transformers import AutoProcessor
        from vjev import Decider
        from vjev.backend import QwenVLBackend
        self.decider = Decider(QwenVLBackend(model), preset=preset)
        self.small = AutoProcessor.from_pretrained(model, min_pixels=hold_pixels, max_pixels=hold_pixels)
        self.small.tokenizer.padding_side = "left"
        self.pose_weights, self.hold_crop = pose_weights, hold_crop
        self.embedder_kind, self._embedder = embedder, None
        self.lock = threading.Lock()                 # one GPU, many sessions

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.embedder_kind)
        return self._embedder

    def prepare(self, catalogue: Catalogue):
        """Embed the catalogue photos once per session (products without photos are named by the VLM)."""
        refs = {}
        for p in catalogue.products:
            if p.images:
                imgs = [decode_image(src) for src in p.images[:5]]
                with self.lock:
                    refs[p.sku] = self.embedder(imgs)
        catalogue.refs = refs

    def new_tracker(self):
        return PoseTracker(self.pose_weights)

    def _yes_no(self, images, question) -> np.ndarray:
        be = self.decider.backend
        content = [{"type": "image"}, {"type": "text", "text": question + " Answer Yes or No."}]
        prompt = self.small.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        inp = self.small(text=[prompt] * len(images), images=images, return_tensors="pt", padding=True).to(be.device)
        with be.torch.no_grad():
            logits = be.model(**inp).logits[:, -1].float()
        z = (logits[:, be.token_id("Yes")] - logits[:, be.token_id("No")]).cpu().numpy()
        return np.asarray(self.decider.cal.noul_platt(z[:, None]), dtype=float).reshape(-1)

    def hold_crops(self, frame, tracks):
        crops, owner = [], []
        for i, tr in enumerate(tracks):
            regions = hand_regions(tr) if self.hold_crop == "hands" else {}
            if regions:
                for r in regions.values():
                    crops.append((crop(frame, r), HAND_QUESTION))
                    owner.append(i)
            else:
                crops.append((crop(frame, person_region(tr.box)), HOLD_QUESTION))
                owner.append(i)
        return crops, owner

    def p_hold(self, frame, tracks) -> List[float]:
        if not tracks:
            return []
        crops, owner = self.hold_crops(frame, tracks)
        p = np.zeros(len(crops))
        with self.lock:
            for q in (HAND_QUESTION, HOLD_QUESTION):
                idx = [k for k, (_, qq) in enumerate(crops) if qq == q]
                if idx:
                    p[idx] = self._yes_no([crops[k][0] for k in idx], q)
        out = [0.0] * len(tracks)
        for k, i in enumerate(owner):
            out[i] = max(out[i], float(p[k]))
        return out

    def p_hand(self, crops) -> List[float]:
        if not crops:
            return []
        with self.lock:
            return [float(p) for p in self._yes_no(crops, HAND_QUESTION)]

    def stock(self, crops) -> List[dict]:
        """Shelf watch: a calibrated distribution over STOCK_LEVELS for each zone crop."""
        out = []
        with self.lock:
            for c in crops:
                a = self.decider.decide({"assets": {"z": c}, "questions": {
                    "s": {"type": "score", "question": STOCK_QUESTION, "levels": STOCK_LEVELS, "image": "z"}}})
                out.append(a["answers"]["s"]["distribution"])
        return out

    def identify(self, frame, track: Track, catalogue: Catalogue, region=None) -> dict:
        if len(catalogue) > Catalogue.MAX_OPTIONS:
            raise ValueError(f"choice identification supports up to {Catalogue.MAX_OPTIONS} products")
        h, w = frame.shape[:2]
        names = [p.name for p in catalogue.products]
        refs = getattr(catalogue, "refs", {})
        skus = [p.sku for p in catalogue.products]
        if region is not None and refs and all(s in refs for s in skus):   # image match against catalogue photos
            x0, y0, x1, y1 = clamp_box(region, w, h)
            with self.lock:
                q = self.embedder([crop(frame, region)])[0]
            sku, conf, probs = rank_products(q, refs, skus, self.embedder.temperature)
            return {"sku": sku, "confidence": conf, "method": f"image-match/{self.embedder.kind}", "probabilities": probs,
                    "box": [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)],
                    "box_confidence": 1.0}
        if region is not None:                       # the hand is known: no grounding pass needed
            x0, y0, x1, y1 = clamp_box(region, w, h)
            with self.lock:
                what = self.decider.decide({"assets": {"c": crop(frame, region)}, "questions": {
                    "w": {"type": "choice", "question": "Which product is in this hand?", "options": names,
                          "image": "c"}}})["answers"]["w"]
            product = catalogue.products[names.index(what["choice"])]
            return {"sku": product.sku, "confidence": what["confidence"],
                    "box": [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)],
                    "box_confidence": 1.0}
        img, (cx, cy, cw, ch) = crop_person(frame, track.box)
        with self.lock:
            box = self.decider.decide({"assets": {"c": img}, "questions": {
                "b": {"type": "box", "target": PRODUCT_TARGET, "image": "c"}}})["answers"]["b"]
            target = img
            if not box["abstained"] and box.get("box_confidence", 0) >= 0.5:
                target = _expand(img, box["box"], 0.3)
            what = self.decider.decide({"assets": {"c": target}, "questions": {
                "w": {"type": "choice", "question": PRODUCT_QUESTION, "options": names, "image": "c"}}})["answers"]["w"]
        product = catalogue.products[names.index(what["choice"])]
        fb, bc = None, 0.0
        if not box["abstained"]:
            b = box["box"]
            fb = [round((cx + b["left"] * cw) / w, 4), round((cy + b["top"] * ch) / h, 4),
                  round((cx + b["right"] * cw) / w, 4), round((cy + b["bottom"] * ch) / h, 4)]
            bc = box.get("box_confidence", 0.0)
        return {"sku": product.sku, "confidence": what["confidence"], "box": fb, "box_confidence": bc}


class ModelUnavailable(RuntimeError):
    """A hosted model did not answer. The caller abstains (a visit stays unchecked, a line stays to verify)."""


NEMOTRON_VL = "nvidia/Nemotron-3.5-Super-VL"


class NemotronEngine(VjevEngine):
    """The VjevEngine funnel with the vision decisions on NVIDIA Nemotron VL via Nebius Token Factory.

    Local, every frame: YOLO11n-pose + ByteTrack (no API call per frame). Hosted, only when a hand
    visits a shelf zone: the before/after "holding a product?" check and shelf-watch readings, through
    vjev's TokenFactoryBackend (calibrated). Naming: SigLIP photo match stays first (local, fast);
    Nemotron names products without a photo and gives a second opinion when SigLIP's top two are close.
    No GPU is needed for the vision-language part."""

    label = "Nemotron-3.5-Super-VL via Nebius Token Factory"

    def __init__(self, preset: str = "fast", pose_weights: str = "yolo11n-pose.pt", embedder: str = "siglip",
                 model: str = NEMOTRON_VL, backend=None, second_opinion_margin: float = 0.15, calibration=None,
                 naming: str = "auto"):
        from vjev import Decider
        from vjev.tokenfactory import TokenFactoryBackend
        self.backend = backend or TokenFactoryBackend(model)
        self.decider = Decider(self.backend, calibration=calibration or nemotron_calibration(), preset=preset)
        self.pose_weights, self.hold_crop = pose_weights, "hands"
        self.embedder_kind, self._embedder = embedder, None
        self.second_opinion_margin = second_opinion_margin
        if naming == "auto":                         # SigLIP takes ~8 s per take on a 4-vCPU box: name by Nemotron there
            try:
                import torch
                naming = "siglip" if torch.cuda.is_available() else "nemotron"
            except ImportError:
                naming = "nemotron"
        if naming not in ("siglip", "nemotron"):
            raise ValueError("naming must be auto, siglip or nemotron")
        self.naming = naming
        self.pose_imgsz = 640 if naming == "siglip" else 416     # CPU: 17 fps at 416 px vs 10 fps at 640
        self.lock = threading.Lock()                 # guards the local embedder; API calls run concurrently

    def new_tracker(self):
        return PoseTracker(self.pose_weights, imgsz=self.pose_imgsz)

    def prepare(self, catalogue: Catalogue):
        if self.naming == "nemotron":                # no local embedder: products are named by Nemotron VL
            catalogue.refs = {}
            return
        super().prepare(catalogue)

    def _yes_no(self, images, question) -> np.ndarray:
        from vjev.tokenfactory import BackendUnavailable
        q = question + " Answer Yes or No."
        try:
            zs = self.backend.batch_next_token_logits([(im, q, ["Yes", "No"]) for im in images])
        except BackendUnavailable as e:
            raise ModelUnavailable(str(e)) from e
        z = np.array([float(a[0] - a[1]) for a in zs])
        return np.asarray(self.decider.cal.noul_platt(z[:, None]), dtype=float).reshape(-1)

    def p_hold(self, frame, tracks) -> List[float]:
        crops, owner = self.hold_crops(frame, tracks)
        p = np.zeros(len(crops))
        for q in (HAND_QUESTION, HOLD_QUESTION):
            idx = [k for k, (_, qq) in enumerate(crops) if qq == q]
            if idx:
                p[idx] = self._yes_no([crops[k][0] for k in idx], q)
        out = [0.0] * len(tracks)
        for k, i in enumerate(owner):
            out[i] = max(out[i], float(p[k]))
        return out

    def p_hand(self, crops) -> List[float]:
        return [float(p) for p in self._yes_no(crops, HAND_QUESTION)] if crops else []

    def stock(self, crops) -> List[dict]:
        from vjev.tokenfactory import BackendUnavailable
        out = []
        for c in crops:
            try:
                a = self.decider.decide({"assets": {"z": c}, "questions": {
                    "s": {"type": "score", "question": STOCK_QUESTION, "levels": STOCK_LEVELS, "image": "z"}}})
            except BackendUnavailable as e:
                raise ModelUnavailable(str(e)) from e
            out.append(a["answers"]["s"]["distribution"])
        return out

    def identify(self, frame, track: Track, catalogue: Catalogue, region=None) -> dict:
        from vjev.tokenfactory import BackendUnavailable
        try:
            res = super().identify(frame, track, catalogue, region)
        except BackendUnavailable as e:
            raise ModelUnavailable(str(e)) from e
        probs = res.get("probabilities") or {}
        if not res.get("method", "").startswith("image-match") or region is None or len(probs) < 2:
            res.setdefault("method", f"nemotron-choice/{self.backend.model}")
            return res
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        if ranked[0][1] - ranked[1][1] >= self.second_opinion_margin:
            return res
        top = [s for s, _ in ranked[:3]]                 # too close to call: ask Nemotron among the top three
        names = [catalogue.by_sku[s].name for s in top]
        try:
            what = self.decider.decide({"assets": {"c": crop(frame, region)}, "questions": {
                "w": {"type": "choice", "question": "Which product is in this hand?", "options": names,
                      "image": "c"}}})["answers"]["w"]
        except BackendUnavailable:
            res["second_opinion"] = {"model": self.backend.model, "unavailable": True}
            res["confidence"] = min(res["confidence"], 0.45)          # unresolved: leave it for the exit check
            return res
        vl_sku = top[names.index(what["choice"])]
        res["second_opinion"] = {"model": self.backend.model, "sku": vl_sku, "confidence": what["confidence"],
                                 "agrees": vl_sku == res["sku"]}
        if vl_sku == res["sku"]:
            res["confidence"] = max(res["confidence"], what["confidence"])
        else:                                            # they disagree: keep SigLIP's pick, flag for the exit check
            res["confidence"] = min(res["confidence"], 0.45)
        res["method"] = res["method"] + "+nemotron"
        return res

    def trace(self) -> dict:
        c = self.backend.client
        return {"engine": self.label, "naming": self.naming, "usage": dict(c.usage), "calls": list(c.trace)[-50:],
                "logprobs": self.backend.logprobs_supported}


def nemotron_calibration():
    """vjev calibration for Nemotron VL: fitted Platt/temperatures if a file is shipped, else identity."""
    import json
    import os
    from vjev import Calibration
    from vjev.calibration import Logistic
    cal = Calibration(choice_temperature=1.0, score_temperature=1.0)
    cal.noul_platt = Logistic([0.0], [1.0], [1.0, 0.0])
    path = os.environ.get("NEMOTRON_CALIBRATION", os.path.join(os.path.dirname(__file__), "nemotron_calibration.json"))
    if os.path.exists(path):
        d = json.load(open(path))
        cal.noul_platt = Logistic.from_dict(d["noul_platt"])
        cal.choice_temperature = float(d.get("choice_temperature", 1.0))
        cal.score_temperature = float(d.get("score_temperature", 1.0))
    return cal


def _expand(img, b, pad):
    w, h = img.size
    x0, y0, x1, y1 = b["left"] * w, b["top"] * h, b["right"] * w, b["bottom"] * h
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return img.crop((max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)))


class StubEngine:
    """Scripted perception: one shopper in the middle of the frame, P(hold) = hold_fn(t),
    and the first catalogue product (or `sku`) at `confidence`. The frame's timestamp is
    passed through `frame_time`, set by the pipeline before each call."""

    def __init__(self, hold_fn: Callable[[float], float], sku: Optional[str] = None, confidence: float = 0.9,
                 identify_s: float = 0.0, hand_fn: Optional[Callable[[float], Tuple[float, float]]] = None,
                 stock_fn: Optional[Callable[[float], str]] = None):
        self.hold_fn, self.sku, self.confidence, self.identify_s = hold_fn, sku, confidence, identify_s
        self.stock_fn = stock_fn or (lambda t: "full")                # shelf-watch label at time t
        self.hand_fn = hand_fn or (lambda t: (0.5, 0.5))          # hand position at time t (frame fractions)
        self.frame_time = 0.0
        self.lock = threading.Lock()

    def new_tracker(self):
        def track(frame):
            h, w = frame.shape[:2]
            kp = np.zeros((17, 3))
            hx, hy = self.hand_fn(self.frame_time)
            kp[R_ELBOW] = (hx * w - 0.05 * w, hy * h, 0.9)        # forearm pointing right
            kp[R_WRIST] = (hx * w - 0.35 / 1.35 * 0.05 * w, hy * h, 0.9)
            return [Track(1, (w * 0.3, h * 0.1, w * 0.7, h * 0.95), 0.9, kp)]
        return track

    def p_hold(self, frame, tracks):
        return [self.hold_fn(self.frame_time) for _ in tracks]

    def p_hand(self, crops):
        """crops carry the time they were taken (the pipeline passes images; tests read .info)."""
        return [self.hold_fn(c.info.get("t", self.frame_time)) for c in crops]

    def stock(self, crops):
        return [{lv: (1.0 if lv == self.stock_fn(c.info.get("t", self.frame_time)) else 0.0) for lv in STOCK_LEVELS}
                for c in crops]

    def identify(self, frame, track, catalogue, region=None):
        if self.identify_s:
            import time
            time.sleep(self.identify_s)
        sku = self.sku or catalogue.products[0].sku
        return {"sku": sku, "confidence": self.confidence, "box": [0.4, 0.4, 0.6, 0.6], "box_confidence": 0.8}
