import pytest
from fastapi.testclient import TestClient

from aisle_agent.layout import Layout, validate
from aisle_agent.models import StubEngine
from aisle_agent.server import create_app

SHOP = {"name": "Corner shop",
        "products": [{"sku": "milk", "name": "carton of milk", "price": 68}, {"sku": "jam", "name": "jar of jam", "price": 189}],
        "shelves": [{"name": "Dairy fridge", "kind": "fridge", "x": 2, "y": 2, "w": 6, "h": 2, "zone": "Dairy",
                     "items": [{"sku": "milk", "qty": 3, "min": 1}]},
                    {"name": "Aisle 1", "x": 2, "y": 8, "w": 10, "h": 2, "items": [{"sku": "jam", "qty": 5, "min": 2}]}]}


def test_validate_rejects_bad_layouts():
    assert validate(SHOP)["shelves"][0]["id"] == "dairy-fridge"
    for bad in ({**SHOP, "shelves": [{**SHOP["shelves"][0], "x": 39}]},                       # off the plan
                {**SHOP, "shelves": [{**SHOP["shelves"][0], "items": [{"sku": "nope", "qty": 1}]}]},
                {**SHOP, "products": SHOP["products"] * 2},
                {**SHOP, "shelves": [{**SHOP["shelves"][0], "kind": "rocket"}]}):
        with pytest.raises(ValueError):
            validate(bad)


def test_stock_follows_takes_exit_and_alerts(tmp_path):
    lay = Layout(str(tmp_path / "l.json"))
    lay.replace(SHOP)
    take = {"type": "line", "merged": False, "line": {"id": 1, "sku": "milk", "zone": "dairy"}}   # zone match ignores case
    lay.observe("s1", [take])
    lay.observe("s1", [{**take, "merged": True}])                       # a re-grab is not a second carton
    lay.observe("s1", [{**take, "line": {"id": 2, "sku": "milk", "zone": "Dairy"}}])
    v = lay.view()
    assert v["shelves"][0]["items"][0]["qty"] == 1 and v["shelves"][0]["state"] == "low"
    lay.observe("s1", [{"type": "review", "line": {"id": 2, "sku": "milk", "status": "removed"}}])
    assert lay.view()["shelves"][0]["items"][0]["qty"] == 2
    lay.observe("s1", [{"type": "restock", "zone": "Aisle 1", "level": 0.2}])   # no zone set: matched by shelf name
    assert lay.view()["shelves"][1]["state"] == "alert"
    lay.replace(SHOP)                                                    # editing keeps the live alert
    assert lay.view()["shelves"][1]["state"] == "alert"
    v = lay.restock("aisle-1", "jam", 12)
    assert v["shelves"][1]["state"] == "ok" and v["shelves"][1]["items"][0]["qty"] == 12
    assert Layout(str(tmp_path / "l.json")).view()["shelves"][1]["items"][0]["qty"] == 12      # saved
    assert lay.catalogue()["zones"] == {"Dairy": ["milk"], "Aisle 1": ["jam"]}


def test_layout_api(monkeypatch):
    monkeypatch.delenv("AISLE_KEY", raising=False)
    with TestClient(create_app(StubEngine(lambda t: 0.1))) as c:
        assert c.get("/layout").status_code == 200
        assert c.put("/v1/layout", json={**SHOP, "shelves": [{"name": "x", "w": 99}]}).status_code == 400
        v = c.put("/v1/layout", json=SHOP).json()
        assert v["value"] == 3 * 68 + 5 * 189 and [s["state"] for s in v["shelves"]] == ["ok", "ok"]
        assert c.post("/v1/layout/shelves/nope/restock", json={}).status_code == 404
        assert c.post("/v1/layout/shelves/aisle-1/restock", json={"sku": "jam", "qty": -1}).status_code == 400
        assert c.get("/v1/layout/catalogue").json()["zones"]["Dairy"] == ["milk"]
