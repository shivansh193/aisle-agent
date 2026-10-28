"""One live stream = one Session: frames in, events out.

Two ways to find takes:

zones (recommended; the store's planogram drawn on the camera image)
  1. track people and their hands (every frame, cheap, no VLM)
  2. a hand enters a shelf zone and leaves it again -> a visit
  3. `after_delay` s after it leaves: one batched check, "is this hand holding a product?",
     on the hand just before it entered and just after it left
       empty -> holding = take (bill line, product named among that zone's products only)
       holding -> empty = put-back (the shopper's matching line is flagged for the exit check)
       otherwise        = touch (no bill change)
  The VLM runs only when a hand actually visits a shelf.

hold (no zones configured; any camera, no setup)
  P(holding a product) per shopper at `hold_hz`, a streaming take detector, then the product.
  Weaker: a shopper who already holds something hides their next take.

Model work after the frame loop (checks, naming) runs in the background unless
`async_identify` is off. Frames that arrive while one is processed are dropped (newest wins).
"""
import asyncio
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from .logic import Bill, Catalogue, HandVisits, StockWatch, TakeDetector, Zone, classify_visit
from .models import ModelUnavailable, crop, crop_person, hand_regions


def data_uri(img, size: int = 256) -> Optional[str]:
    """Small JPEG data URI of a PIL image (exit-check evidence)."""
    if img is None:
        return None
    import base64
    import io
    im = img.convert("RGB").copy()
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@dataclass
class Options:
    max_hz: float = 10.0
    # hold mode
    hold_hz: float = 5.0
    identify_delay: float = 0.5
    hi: float = 0.6
    lo: float = 0.4
    confirm: float = 0.4
    # zones mode
    after_delay: float = 0.3           # check the hand this long after it leaves the shelf
    min_dwell: float = 0.15
    near_margin: float = 0.12          # keep a "before" crop while the hand is this close to a zone
    hold_yes: float = 0.6
    hold_no: float = 0.5
    watch_every: float = 5.0           # shelf watch: read each zone's stock this often (0 = off)
    restock_below: float = 0.58        # chosen on half of Retail Shelf Void crops; 82% on the held-out half
    # both
    verify_below: float = 0.5
    merge_within: float = 6.0                  # a repeat take of the same product by the same hand is one item
    forget_after: float = 5.0
    async_identify: bool = True


class Session:
    def __init__(self, sid: str, engine, catalogue: Catalogue, options: Optional[Options] = None, currency="INR",
                 zones: Optional[List[dict]] = None):
        self.id, self.engine, self.catalogue = sid, engine, catalogue
        self.opt = options or Options()
        self.tracker = engine.new_tracker()
        self.bill = Bill(currency, self.opt.verify_below, self.opt.merge_within)
        if hasattr(engine, "prepare"):
            engine.prepare(catalogue)                  # e.g. embed catalogue photos
        self.set_zones(zones or [])
        self.detectors: Dict[int, TakeDetector] = {}
        self.last_hold: Dict[int, float] = {}
        self.last_seen: Dict[int, tuple] = {}          # id -> (t, Track)
        self.p_now: Dict[int, float] = {}
        self.before: Dict[tuple, tuple] = {}           # (shopper, hand) -> (t, crop) last seen just outside a zone
        self.pending: List[dict] = []
        self.evidence: Dict[int, dict] = {}            # line id -> what the exit check is shown
        self.preview = None                            # (t, jpeg bytes, width, height) of the latest analysed frame
        self.stock = StockWatch(low=self.opt.restock_below, ok=min(0.95, self.opt.restock_below + 0.12))
        self.last_watch: Dict[str, float] = {}
        self.subscribers: set = set()
        self.listeners: list = []                 # server-side observers (shop layout stock)
        self.timing = defaultdict(lambda: deque(maxlen=200))
        self.counts = defaultdict(int)
        self.t0 = time.monotonic()
        self.latest = None
        self._new = asyncio.Event()
        self._worker: Optional[asyncio.Task] = None
        self._bg: set = set()
        self.closed = False

    def set_zones(self, zones: List[dict]):
        parsed = [Zone(**z) for z in zones]
        for z in parsed:
            l, t, r, b = z.box
            if not (0 <= l < r <= 1 and 0 <= t < b <= 1):
                raise ValueError(f"zone {z.id}: box must be [left, top, right, bottom] fractions")
            unknown = [s for s in z.skus if s not in self.catalogue.by_sku]
            if unknown:
                raise ValueError(f"zone {z.id}: unknown skus {unknown}")
        self.zones = parsed
        self.visits = HandVisits(parsed, self.opt.min_dwell)

    @property
    def mode(self) -> str:
        return "zones" if self.zones else "hold"

    # ------------------------------------------------------------------ core (sync, runs in a thread)

    def process(self, frame: np.ndarray, t: float):
        """One frame -> (events, model jobs that are now due)."""
        msgs, jobs = [], []
        h, w = frame.shape[:2]
        if hasattr(self.engine, "frame_time"):
            self.engine.frame_time = t
        s = time.perf_counter()
        tracks = self.tracker(frame)
        self._time("track_ms", s)
        for tr in tracks:
            self.last_seen[tr.id] = (t, tr)

        hands_out = []
        if self.mode == "zones":
            hands_out = self._zones_step(frame, t, tracks, msgs)
        else:
            self._hold_step(frame, t, tracks, msgs)

        for job in [j for j in self.pending if t >= j["at"]]:
            self.pending.remove(job)
            if job["kind"] == "watch":
                jobs.append(dict(job, t=t))
                continue
            seen = self.last_seen.get(job["shopper"])
            if seen is None or t - seen[0] > 1.0:
                if job["kind"] == "visit":
                    self.counts["visits_lost"] += 1
                continue
            job = dict(job, frame=frame, track=seen[1], t=t)
            if job["kind"] == "visit":
                region = hand_regions(seen[1]).get(job["hand"])
                if region is None:
                    self.counts["visits_lost"] += 1
                    continue
                job["region"] = region
                job["after"] = self._crop(frame, region, t)
            jobs.append(job)

        for sid in [k for k, (ts, _) in self.last_seen.items() if t - ts > self.opt.forget_after]:
            for d in (self.last_seen, self.detectors, self.last_hold, self.p_now):
                d.pop(sid, None)
            for key in [k for k in self.before if k[0] == sid]:
                del self.before[key]

        self.counts["processed"] += 1
        self._set_preview(frame, t)
        msgs.append({"type": "state", "t": round(t, 3), "shoppers": [
            {"id": tr.id, "box": [round(tr.box[0] / w, 4), round(tr.box[1] / h, 4), round(tr.box[2] / w, 4),
                                  round(tr.box[3] / h, 4)],
             "p_hold": round(self.p_now.get(tr.id, 0.0), 3)} for tr in tracks], "hands": hands_out})
        return msgs, jobs

    def _crop(self, frame, region, t):
        img = crop(frame, region)
        img.thumbnail((448, 448))
        img.info["t"] = t
        return img

    def _zones_step(self, frame, t, tracks, msgs):
        h, w = frame.shape[:2]
        hands, regions, out, near = {}, {}, [], {}
        for tr in tracks:
            for k, r in hand_regions(tr).items():
                x, y = (r[0] + r[2]) / 2 / w, (r[1] + r[3]) / 2 / h
                hands[(tr.id, k)] = (x, y)
                regions[(tr.id, k)] = r
                z = self.visits.zone_at(x, y)
                out.append({"shopper": tr.id, "hand": k, "zone": z.id if z else None,
                            "box": [round(r[0] / w, 4), round(r[1] / h, 4), round(r[2] / w, 4), round(r[3] / h, 4)]})
                if z is None and any(zz.contains(x, y, self.opt.near_margin) for zz in self.zones):
                    near[(tr.id, k)] = r
        for v in self.visits.update(t, hands):
            shopper, hand = v["key"]
            b = self.before.get(v["key"])
            before = b[1] if b is not None and b[0] <= v["t_enter"] + 1e-6 and v["t_enter"] - b[0] < 2.0 else None
            self.counts["visits"] += 1
            msgs.append({"type": "visit", "shopper": shopper, "hand": hand, "zone": v["zone"],
                         "t_enter": round(v["t_enter"], 3), "t_exit": round(v["t_exit"], 3)})
            self.pending.append({"kind": "visit", "shopper": shopper, "hand": hand, "zone": v["zone"],
                                 "t_enter": v["t_enter"], "rise": v["t_exit"], "before": before,
                                 "at": v["t_exit"] + self.opt.after_delay})
        for key, r in near.items():                    # after the visits, so an exit frame is never its own "before"
            self.before[key] = (t, self._crop(frame, r, t))
        if self.opt.watch_every > 0:                   # shelf watch, only when no hand is in the zone
            busy = {o["zone"] for o in out if o["zone"]}
            for z in self.zones:
                if z.id not in busy and t - self.last_watch.get(z.id, -1e9) >= self.opt.watch_every:
                    self.last_watch[z.id] = t
                    l, tp, r, b = z.box
                    self.pending.append({"kind": "watch", "zone": z.id, "shopper": None, "at": t,
                                         "crop": self._crop(frame, (l * w, tp * h, r * w, b * h), t)})
        return out

    def _hold_step(self, frame, t, tracks, msgs):
        o = self.opt
        due = [tr for tr in tracks if t - self.last_hold.get(tr.id, -1e9) >= 1.0 / o.hold_hz - 1e-6]
        if not due:
            return
        s = time.perf_counter()
        ps = self.engine.p_hold(frame, due)
        self._time("hold_ms", s)
        for tr, p in zip(due, ps):
            self.last_hold[tr.id] = t
            det = self.detectors.setdefault(tr.id, TakeDetector(o.hi, o.lo, o.confirm))
            rise = det.update(t, float(p))
            self.p_now[tr.id] = det.p
            if rise is not None:
                self.counts["takes"] += 1
                msgs.append({"type": "take", "shopper": tr.id, "t": round(rise, 3), "detected_at": round(t, 3)})
                self.pending.append({"kind": "take", "shopper": tr.id, "rise": rise, "at": t + o.identify_delay})

    def resolve(self, job) -> List[dict]:
        """Model work for one due job (sync; may run in a thread): visit check and/or naming."""
        out = []
        catalogue, region, zone, proof = self.catalogue, None, None, {}
        if job["kind"] == "watch":
            s = time.perf_counter()
            dist = self.engine.stock([job["crop"]])[0]
            self._time("watch_ms", s)
            out = self.stock.update(job["zone"], job["t"], dist)
            self.counts["restock_alerts"] += sum(1 for e in out if e["type"] == "restock")
            return out
        if job["kind"] == "visit":
            s = time.perf_counter()
            crops = ([job["before"]] if job["before"] is not None else []) + [job["after"]]
            try:
                ps = self.engine.p_hand(crops)
            except ModelUnavailable as e:                # hosted model down: say so, never guess a take
                self.counts["unchecked"] += 1
                return [{"type": "unchecked", "shopper": job["shopper"], "zone": job["zone"],
                         "t": round(job["rise"], 3), "reason": str(e)[:120]}]
            self._time("check_ms", s)
            pb, pa = (ps[0], ps[1]) if job["before"] is not None else (None, ps[0])
            kind = classify_visit(pb, pa, self.opt.hold_yes, self.opt.hold_no)
            self.counts[kind + "s"] += 1
            zone = next(z for z in self.zones if z.id == job["zone"])
            ev = {"type": kind, "shopper": job["shopper"], "hand": job["hand"], "zone": zone.id,
                  "t": round(job["rise"], 3), "p_before": None if pb is None else round(pb, 3), "p_after": round(pa, 3)}
            proof = {"kind": kind, "zone": zone.id, "t": ev["t"], "p_before": ev["p_before"], "p_after": ev["p_after"],
                     "before": data_uri(job["before"]), "after": data_uri(job["after"])}
            if kind == "put":
                line = self.bill.flag_put_back(job["shopper"], zone.id, zone.skus)
                ev["flagged_line"] = line.id if line else None
                if line:
                    self.evidence.setdefault(line.id, {})["put_back"] = proof
                out += [ev, {"type": "bill", "bill": self.bill.as_dict()}] if line else [ev]
                return out
            out.append(ev)
            if kind != "take":
                return out
            catalogue, region = catalogue.subset(zone.skus) if zone.skus else catalogue, job["region"]
        s = time.perf_counter()
        res = self.engine.identify(job["frame"], job["track"], catalogue, region)
        self._time("identify_ms", s)
        box = res.get("box") if res.get("box_confidence", 0) >= 0.5 else None
        seen = {l.id: l.seen for l in self.bill.lines.values()}
        line = self.bill.add(self.catalogue.by_sku[res["sku"]], res["confidence"], job["shopper"], job["rise"], box,
                             zone.id if zone else None)
        merged = line.id in seen
        done = job["t"] + (time.perf_counter() - s)
        if not proof:                                   # hold mode: the shopper at the moment of naming
            proof = {"kind": "take", "t": round(job["rise"], 3), "after": data_uri(crop_person(job["frame"], job["track"].box)[0])}
        if merged:
            self.counts["merged"] += 1
            self.evidence.setdefault(line.id, {}).setdefault("repeats", []).append(proof)
        else:
            self.evidence[line.id] = {"take": proof, "method": res.get("method", "vlm-choice"),
                                      "candidates": res.get("probabilities"), "confidence": res["confidence"]}
        out.append({"type": "line", "merged": merged, "line": vars(line).copy(), "latency_s": round(done - job["rise"], 3),
                    "method": res.get("method", "vlm-choice"), "candidates": res.get("probabilities"),
                    "bill": self.bill.as_dict()})
        return out

    def _set_preview(self, frame, t, width: int = 960):
        """Keep the latest analysed frame as a small JPEG for the live-feed endpoint (any source)."""
        import cv2
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (width, int(h * width / w))) if w > width else frame
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            self.preview = (t, buf.tobytes(), w, h)

    def _time(self, key, start, per=1):
        self.timing[key].append((time.perf_counter() - start) * 1000 / max(per, 1))

    # ------------------------------------------------------------------ async side

    def now(self) -> float:
        return time.monotonic() - self.t0

    async def handle(self, frame: np.ndarray, t: Optional[float] = None) -> List[dict]:
        """Process one frame now and publish its events. With async work, the results of
        checks / naming are published on the events stream when ready (not returned here)."""
        msgs, jobs = await asyncio.to_thread(self.process, frame, self.now() if t is None else t)
        for job in jobs:
            if self.opt.async_identify:
                task = asyncio.get_running_loop().create_task(self._resolve_async(job))
                self._bg.add(task)
                task.add_done_callback(self._bg.discard)
            else:
                for m in await asyncio.to_thread(self.resolve, job):
                    msgs.insert(len(msgs) - 1, m)
        self.publish(msgs)
        return msgs

    async def _resolve_async(self, job):
        try:
            self.publish(await asyncio.to_thread(self.resolve, job))
        except Exception as e:
            self.counts["errors"] += 1
            self.publish([{"type": "error", "message": f"{job['kind']}: {type(e).__name__}: {e}"}])

    async def drain(self):
        """Wait for background model work (tests, replay)."""
        while self._bg:
            await asyncio.gather(*list(self._bg), return_exceptions=True)

    def offer(self, frame: np.ndarray, t: Optional[float] = None):
        """Hand a live frame to the worker; an unprocessed older frame is dropped."""
        if self.latest is not None:
            self.counts["dropped"] += 1
        self.counts["received"] += 1
        self.latest = (frame, self.now() if t is None else t)
        self._new.set()
        if self._worker is None:
            self._worker = asyncio.get_running_loop().create_task(self._run())

    async def _run(self):
        period = 1.0 / self.opt.max_hz
        while not self.closed:
            await self._new.wait()
            self._new.clear()
            if self.latest is None:
                continue
            frame, t = self.latest
            self.latest = None
            start = time.monotonic()
            try:
                await self.handle(frame, t)
            except Exception as e:                       # keep the stream alive; report the error
                self.counts["errors"] += 1
                self.publish([{"type": "error", "message": f"{type(e).__name__}: {e}"}])
            rest = period - (time.monotonic() - start)
            if rest > 0:
                await asyncio.sleep(rest)

    def publish(self, msgs: List[dict]):
        for fn in self.listeners:
            try:
                fn(self.id, msgs)
            except Exception:                         # an observer must never stop the stream
                pass
        for q in list(self.subscribers):
            for m in msgs:
                if q.qsize() < 500:
                    q.put_nowait(m)

    def stats(self) -> dict:
        return {"id": self.id, "mode": self.mode, "uptime_s": round(self.now(), 1), "counts": dict(self.counts),
                "shoppers_tracked": len(self.last_seen), "options": asdict(self.opt),
                "zones": [asdict(z) for z in self.zones], "shelves": self.stock.as_dict(),
                "timing_ms": {k: {"mean": round(float(np.mean(v)), 1), "p90": round(float(np.percentile(v, 90)), 1)}
                              for k, v in self.timing.items() if v}}

    async def close(self):
        self.closed = True
        self._new.set()
        if self._worker:
            self._worker.cancel()
