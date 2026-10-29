import importlib.util
import json
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location("idle_stop", Path(__file__).parents[1] / "deploy/aws/idle_stop.py")
idle_stop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(idle_stop)


def test_polls_count_as_requests_but_not_page_loads(tmp_path, monkeypatch):
    now = time.time()
    log = tmp_path / "access.log"
    rows = [(now - 4000, "GET", "/"), (now - 3500, "GET", "/b/abc"), (now - 60, "GET", "/v1/audit"),
            (now - 30, "POST", "/layout")]
    log.write_text("\n".join(json.dumps({"ts": t, "request": {"method": m, "uri": u}}) for t, m, u in rows) + "\nnot json\n")
    monkeypatch.setattr(idle_stop, "LOG", str(log))
    monkeypatch.setattr(idle_stop, "booted_at", lambda: now - 5000)
    last_any, last_page = idle_stop.last_activity()
    assert abs(last_any - (now - 30)) < 1
    assert abs(last_page - (now - 3500)) < 1


def test_no_log_counts_from_boot(tmp_path, monkeypatch):
    monkeypatch.setattr(idle_stop, "LOG", str(tmp_path / "missing.log"))
    monkeypatch.setattr(idle_stop, "booted_at", lambda: 123.0)
    assert idle_stop.last_activity() == (123.0, 123.0)
