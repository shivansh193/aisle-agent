"""Model-free core: the streaming take detector, the catalogue and the bill.

Kept free of model and web dependencies so it can be tested anywhere.
"""
import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class TakeDetector:
    """Streaming version of the offline hysteresis used for the demo video.

    P(holding a product) is smoothed with a time-aware EMA. A take fires when the smoothed
    value rises through `hi` and stays above `lo` for `confirm` seconds. A shopper who is
    already holding something at the first reading starts in the holding state (no event),
    and the detector re-arms only once the smoothed value falls to `lo`.
    """

    def __init__(self, hi: float = 0.6, lo: float = 0.4, confirm: float = 0.4, tau: float = 0.15):
        self.hi, self.lo, self.confirm, self.tau = hi, lo, confirm, tau
        self.p: Optional[float] = None
        self.t: Optional[float] = None
        self.state = "idle"                     # idle | candidate | holding
        self.t_rise: Optional[float] = None

    def update(self, t: float, p: float) -> Optional[float]:
        """Feed one reading; returns the rise time of a confirmed take, else None."""
        if self.p is None:
            self.p, self.t = p, t
            self.state = "holding" if p >= self.hi else "idle"
            return None
        a = 1.0 - math.exp(-max(t - self.t, 0.0) / self.tau)
        self.p, self.t = self.p + a * (p - self.p), t
        if self.state == "idle":
            if self.p >= self.hi:
                self.state, self.t_rise = "candidate", t
        elif self.state == "candidate":
            if self.p <= self.lo:
                self.state = "idle"
            elif t - self.t_rise >= self.confirm:
                self.state = "holding"
                return self.t_rise
        elif self.p <= self.lo:
            self.state = "idle"
        return None


@dataclass
class Product:
    sku: str
    name: str
    price: float
    zone: Optional[str] = None                 # planogram zone, used by step 2
    images: List[str] = field(default_factory=list)


class Catalogue:
    MAX_OPTIONS = 26                           # vjev choice limit

    def __init__(self, products: List[dict]):
        if products and isinstance(products[0], Product):
            products = [vars(p) for p in products]
        if not products:
            raise ValueError("catalogue is empty")
        self.products = [Product(**p) for p in products]
        skus = [p.sku for p in self.products]
        if len(set(skus)) != len(skus):
            raise ValueError("duplicate sku in catalogue")
        self.by_sku = {p.sku: p for p in self.products}

    def __len__(self):
        return len(self.products)

    def subset(self, skus: List[str]) -> "Catalogue":
        keep = [p for p in self.products if p.sku in set(skus)]
        if not keep:
            return self
        sub = Catalogue(keep)
        sub.refs = {s: v for s, v in getattr(self, "refs", {}).items() if s in sub.by_sku}
        return sub


@dataclass
class Line:
    id: int
    sku: str
    name: str
    price: float
    confidence: float
    status: str                                # ok | verify | disputed | approved | removed
    shopper: Optional[int]
    t: float
    box: Optional[List[float]] = None          # [left, top, right, bottom], fractions of the frame
    zone: Optional[str] = None
    note: Optional[str] = None
    seen: int = 1                              # takes merged into this line (a re-grab is one item)


class Bill:
    """The system only ever adds or flags lines. A dispute or a seen put-back sends a line to
    the exit check, and only a person there (`review`) approves or removes it. A removed line
    stays on the bill, out of the total, with the reason in the audit log."""

    def __init__(self, currency: str = "INR", verify_below: float = 0.5, merge_within: float = 6.0):
        self.currency, self.verify_below, self.merge_within = currency, verify_below, merge_within
        self.lines: Dict[int, Line] = {}
        self.audit: List[dict] = []
        self._ids = itertools.count(1)

    def add(self, product: Product, confidence: float, shopper: Optional[int], t: float, box=None,
            zone: Optional[str] = None) -> Line:
        same = self.repeat_of(product.sku, shopper, zone, t)
        if same is not None:                   # the same hand re-grabbing: one item, keep the surer naming
            same.seen += 1
            same.t = round(t, 3)
            if confidence > same.confidence:
                same.confidence = round(confidence, 4)
                if same.status == "verify" and confidence >= self.verify_below and same.note is None:
                    same.status = "ok"
            return same
        status = "ok" if confidence >= self.verify_below else "verify"
        line = Line(next(self._ids), product.sku, product.name, product.price, round(confidence, 4), status,
                    shopper, round(t, 3), box, zone)
        self.lines[line.id] = line
        return line

    def repeat_of(self, sku: str, shopper: Optional[int], zone: Optional[str], t: float) -> Optional[Line]:
        """A recent open line for the same product, shopper and zone, within `merge_within` seconds."""
        if self.merge_within <= 0:
            return None
        for line in reversed(list(self.lines.values())):
            if t - line.t > self.merge_within:
                return None
            if line.sku == sku and line.shopper == shopper and line.zone == zone and line.status in ("ok", "verify"):
                return line
        return None

    def dispute(self, line_id: int) -> Line:
        line = self.lines[line_id]
        line.status = "disputed"
        return line

    def flag_put_back(self, shopper: Optional[int], zone: Optional[str], skus: List[str]) -> Optional[Line]:
        """A put-back was seen: flag this shopper's latest matching line for the exit check."""
        for line in reversed(list(self.lines.values())):
            if line.status in ("ok", "verify", "approved") and line.shopper == shopper and (line.zone == zone or line.sku in skus):
                line.status, line.note = "verify", "put-back seen"
                return line
        return None

    def review(self, line_id: int, action: str, by: str = "staff", note: Optional[str] = None,
               t: Optional[float] = None) -> Line:
        if action not in ("approve", "remove"):
            raise ValueError("action must be 'approve' or 'remove'")
        line = self.lines[line_id]
        before = line.status
        line.status = "approved" if action == "approve" else "removed"
        line.note = note or line.note
        self.audit.append({"line": line_id, "action": action, "by": by, "from": before, "note": note, "t": t})
        return line

    def as_dict(self) -> dict:
        lines = [vars(l).copy() for l in self.lines.values()]
        return {"currency": self.currency, "lines": lines,
                "total": round(sum(l.price for l in self.lines.values() if l.status != "removed"), 2),
                "to_verify": [l.id for l in self.lines.values() if l.status in ("verify", "disputed")],
                "audit": list(self.audit)}


@dataclass
class Zone:
    """A shelf area in the camera image (fractions of the frame) and the products stocked there."""
    id: str
    box: List[float]                           # [left, top, right, bottom]
    skus: List[str] = field(default_factory=list)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        l, t, r, b = self.box
        return l - margin <= x <= r + margin and t - margin <= y <= b + margin


class HandVisits:
    """Per hand: when it enters a zone and when it leaves. A visit that lasted at least
    `min_dwell` seconds is returned on exit (or when the hand is lost for `lost_after` s)."""

    def __init__(self, zones: List[Zone], min_dwell: float = 0.15, lost_after: float = 1.0):
        self.zones, self.min_dwell, self.lost_after = zones, min_dwell, lost_after
        self.state: Dict[tuple, dict] = {}

    def zone_at(self, x, y) -> Optional[Zone]:
        return next((z for z in self.zones if z.contains(x, y)), None)

    def update(self, t: float, hands: Dict[tuple, tuple]) -> List[dict]:
        """hands: {(shopper, hand): (x, y)} visible this frame (frame fractions)."""
        done = []
        for key, (x, y) in hands.items():
            st = self.state.setdefault(key, {"zone": None, "seen": t})
            st["seen"] = t
            z = self.zone_at(x, y)
            if st["zone"] is None and z is not None:
                st.update(zone=z.id, t_enter=t)
            elif st["zone"] is not None and (z is None or z.id != st["zone"]):
                if t - st["t_enter"] >= self.min_dwell:
                    done.append({"key": key, "zone": st["zone"], "t_enter": st["t_enter"], "t_exit": t})
                st.update(zone=z.id if z else None, t_enter=t)
        for key, st in list(self.state.items()):
            if key not in hands and t - st["seen"] > self.lost_after:
                if st["zone"] is not None and st["seen"] - st["t_enter"] >= self.min_dwell:
                    done.append({"key": key, "zone": st["zone"], "t_enter": st["t_enter"], "t_exit": st["seen"],
                                 "lost": True})
                del self.state[key]
        return done


def classify_visit(p_before: Optional[float], p_after: Optional[float], hi: float = 0.6, lo: float = 0.5) -> str:
    """take: empty before, holding after. put: holding before, empty after. Otherwise touch.
    A hand first seen inside the zone has no 'before'; it counts as empty."""
    pb = 0.0 if p_before is None else p_before
    if p_after is None:
        return "touch"
    if p_after >= hi and pb < lo:
        return "take"
    if pb >= hi and p_after < lo:
        return "put"
    return "touch"


def rank_products(query, refs: Dict[str, "np.ndarray"], skus: List[str], temperature: float):
    """Image match: each product scores its best reference photo (cosine; vectors normalised),
    softmax over the candidate products at a fitted temperature. -> (sku, confidence, {sku: p})"""
    import numpy as np
    cands = [s for s in skus if s in refs and len(refs[s])]
    if not cands:
        raise ValueError("no candidate product has a reference photo")
    sims = np.array([float((refs[s] @ query).max()) for s in cands]) / temperature
    p = np.exp(sims - sims.max())
    p /= p.sum()
    best = int(p.argmax())
    return cands[best], float(p[best]), {s: round(float(v), 4) for s, v in zip(cands, p)}


STOCK_LEVELS = ["empty", "low", "half full", "full"]


class StockWatch:
    """Shelf watch: a running stock reading per zone and restock alerts with hysteresis.

    Each reading is a distribution over STOCK_LEVELS; its expected value is mapped to 0..1
    (0 = empty, 1 = full). An alert fires after `confirm` consecutive readings below `low`, and
    clears ("restocked") once a reading reaches `ok`."""

    def __init__(self, low: float = 0.58, ok: float = 0.7, confirm: int = 2):
        self.low, self.ok, self.confirm = low, ok, confirm
        self.state: Dict[str, dict] = {}

    def update(self, zone: str, t: float, dist: Dict[str, float]) -> List[dict]:
        level = sum(STOCK_LEVELS.index(k) * p for k, p in dist.items()) / (len(STOCK_LEVELS) - 1)
        st = self.state.setdefault(zone, {"below": 0, "alert": False, "history": []})
        st.update(level=round(level, 3), label=max(dist, key=dist.get), t=round(t, 2))
        st["history"] = (st["history"] + [(round(t, 2), round(level, 3))])[-120:]
        out = [{"type": "stock", "zone": zone, "t": round(t, 2), "level": round(level, 3), "label": st["label"]}]
        st["below"] = st["below"] + 1 if level < self.low else 0
        if not st["alert"] and st["below"] >= self.confirm:
            st["alert"] = True
            out.append({"type": "restock", "zone": zone, "t": round(t, 2), "level": round(level, 3)})
        elif st["alert"] and level >= self.ok:
            st["alert"] = False
            out.append({"type": "restocked", "zone": zone, "t": round(t, 2), "level": round(level, 3)})
        return out

    def as_dict(self) -> Dict[str, dict]:
        return {z: {k: v for k, v in s.items() if k != "below"} for z, s in self.state.items()}
