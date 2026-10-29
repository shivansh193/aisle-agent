"""The shop's layout: a floor plan of shelves, the products on each with stock counts, and the
camera zone that watches each shelf. Saved to a JSON file (AISLE_LAYOUT).

Stock follows the bill: a confirmed take in a zone takes one off the shelf linked to that zone;
a line removed at the exit puts it back. A restock alert from shelf watch marks the shelf until
someone restocks it. A repeat take merged into an existing line changes nothing.
"""
import json
import os
import re
import threading
import time
from typing import Dict, List, Optional

KINDS = ("shelf", "fridge", "freezer", "counter", "entrance")
GRID = (40, 26)                                   # floor plan cells (about 0.5 m each)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "item"


def validate(data: dict) -> dict:
    """Clean a layout sent by the editor; raises ValueError with a readable reason."""
    if not isinstance(data, dict):
        raise ValueError("layout must be an object")
    products, skus = [], set()
    for p in data.get("products", []):
        sku = str(p.get("sku") or slug(str(p.get("name", ""))))
        name = str(p.get("name", "")).strip()
        if not name:
            raise ValueError("every product needs a name")
        if sku in skus:
            raise ValueError(f"duplicate product {sku}")
        price = float(p.get("price", 0))
        if price < 0:
            raise ValueError(f"{name}: price can't be negative")
        skus.add(sku)
        products.append({"sku": sku, "name": name, "price": price})
    shelves, ids = [], set()
    for s in data.get("shelves", []):
        sid = str(s.get("id") or slug(str(s.get("name", ""))))
        if sid in ids:
            raise ValueError(f"duplicate shelf {sid}")
        ids.add(sid)
        kind = s.get("kind", "shelf")
        if kind not in KINDS:
            raise ValueError(f"shelf kind must be one of {', '.join(KINDS)}")
        x, y, w, h = (int(s.get(k, d)) for k, d in (("x", 0), ("y", 0), ("w", 6), ("h", 2)))
        if w < 1 or h < 1 or x < 0 or y < 0 or x + w > GRID[0] or y + h > GRID[1]:
            raise ValueError(f"{s.get('name', sid)} doesn't fit on the floor plan")
        items = []
        for it in s.get("items", []):
            if it.get("sku") not in skus:
                raise ValueError(f"{s.get('name', sid)}: unknown product {it.get('sku')}")
            qty, low = int(it.get("qty", 0)), int(it.get("min", 0))
            if qty < 0 or low < 0:
                raise ValueError("stock and minimum can't be negative")
            items.append({"sku": it["sku"], "qty": qty, "min": low})
        shelves.append({"id": sid, "name": str(s.get("name") or sid), "kind": kind, "x": x, "y": y, "w": w, "h": h,
                        "zone": (str(s["zone"]).strip() or None) if s.get("zone") else None, "items": items,
                        "alert": s.get("alert")})
    return {"name": str(data.get("name") or "My shop"), "grid": list(GRID), "products": products, "shelves": shelves}


class Layout:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.environ.get("AISLE_LAYOUT", "shop_layout.json")
        self.lock = threading.Lock()
        self.data = {"name": "My shop", "grid": list(GRID), "products": [], "shelves": []}
        self.log: List[dict] = []                 # recent stock moves, newest last
        self.taken: Dict[tuple, str] = {}         # (session, line) -> shelf id it came off
        if os.path.exists(self.path):
            self.data = validate(json.load(open(self.path, encoding="utf-8")))

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)

    def replace(self, data: dict) -> dict:
        clean = validate(data)
        with self.lock:
            old = {s["id"]: (s.get("alert"), s.get("findings")) for s in self.data["shelves"]}
            for s in clean["shelves"]:                              # the editor doesn't own live alerts or findings
                s["alert"] = old.get(s["id"], (s.get("alert"), None))[0]
                if old.get(s["id"], (None, None))[1]:
                    s["findings"] = old[s["id"]][1]
            self.data = clean
            self.save()
        return self.view()

    def shelf_for(self, zone: Optional[str]) -> Optional[dict]:
        if not zone:
            return None
        z = zone.strip().lower()
        return next((s for s in self.data["shelves"] if (s["zone"] or s["name"]).strip().lower() == z), None)

    def _move(self, shelf, sku, delta, why, t=None):
        it = next((i for i in shelf["items"] if i["sku"] == sku), None)
        if it is None:
            return False
        it["qty"] = max(0, it["qty"] + delta)
        self.log = (self.log + [{"t": t or time.time(), "shelf": shelf["id"], "sku": sku, "delta": delta, "why": why,
                                 "qty": it["qty"]}])[-200:]
        return True

    def observe(self, session_id: str, msgs: List[dict]):
        """Session events -> stock. Called from Session.publish."""
        changed = False
        with self.lock:
            for m in msgs:
                kind, line = m.get("type"), m.get("line") or {}
                if kind == "line" and not m.get("merged"):
                    shelf = self.shelf_for(line.get("zone"))
                    if shelf and self._move(shelf, line["sku"], -1, "taken"):
                        self.taken[(session_id, line["id"])] = shelf["id"]
                        changed = True
                elif kind == "review" and line.get("status") == "removed":
                    sid = self.taken.pop((session_id, line.get("id")), None)
                    shelf = next((s for s in self.data["shelves"] if s["id"] == sid), None)
                    if shelf and self._move(shelf, line["sku"], +1, "removed at exit"):
                        changed = True
                elif kind == "restock":
                    shelf = self.shelf_for(m.get("zone"))
                    if shelf:
                        shelf["alert"] = {"level": m.get("level"), "t": time.time()}
                        changed = True
                elif kind == "audit":                                   # shelf audit findings on the plan
                    shelf = self.shelf_for(m.get("zone"))
                    if shelf is not None:
                        shelf["findings"] = [{k: f.get(k) for k in ("kind", "item", "product", "label_price", "catalogue_price")}
                                             for f in m.get("findings") or []]
                        changed = True
                elif kind == "restocked":                               # the camera sees it full again
                    shelf = self.shelf_for(m.get("zone"))
                    if shelf and shelf.get("alert"):
                        shelf["alert"] = None
                        changed = True
            if changed:
                self.save()

    def restock(self, shelf_id: str, sku: Optional[str] = None, qty: Optional[int] = None) -> dict:
        """Someone refilled a shelf: set an item's count (or clear just the alert)."""
        with self.lock:
            shelf = next((s for s in self.data["shelves"] if s["id"] == shelf_id), None)
            if shelf is None:
                raise KeyError(shelf_id)
            if sku is not None:
                it = next((i for i in shelf["items"] if i["sku"] == sku), None)
                if it is None or qty is None or qty < 0:
                    raise ValueError("give a product on this shelf and a count of 0 or more")
                self._move(shelf, sku, qty - it["qty"], "restocked")
            shelf["alert"] = None
            self.save()
        return self.view()

    def set_price(self, sku: str, price: float):
        with self.lock:
            for p in self.data["products"]:
                if p["sku"] == sku:
                    p["price"] = round(float(price), 2)
                    self.save()

    def view(self) -> dict:
        """The layout plus each shelf's state: ok | low | empty | alert."""
        names = {p["sku"]: p for p in self.data["products"]}
        out = json.loads(json.dumps(self.data))
        for s in out["shelves"]:
            low = [i for i in s["items"] if i["qty"] <= i["min"]]
            s["state"] = ("empty" if s["items"] and all(i["qty"] == 0 for i in s["items"]) else
                          "alert" if s.get("alert") else "low" if low else "ok")
            s["low"] = [names[i["sku"]]["name"] for i in low if i["sku"] in names]
            s["units"] = sum(i["qty"] for i in s["items"])
            s["findings"] = s.get("findings") or []
        out["value"] = round(sum(i["qty"] * names[i["sku"]]["price"] for s in out["shelves"] for i in s["items"]
                                 if i["sku"] in names), 2)
        out["log"] = self.log[-30:]
        return out

    def catalogue(self) -> dict:
        """For the console: the products, and which products each zone (shelf) holds."""
        return {"products": self.data["products"],
                "zones": {(s["zone"] or s["name"]): [i["sku"] for i in s["items"]] for s in self.data["shelves"]
                          if s["kind"] not in ("counter", "entrance")}}
