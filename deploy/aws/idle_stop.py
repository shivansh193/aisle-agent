"""Stop this instance when nobody is using it. Run every minute as root by aisle-idle-stop.timer.

It shuts down (which stops an EBS-backed EC2 instance, so billing stops) when either:
- there has been no request at all for IDLE_MIN minutes (default 15), or
- no page has been opened for PAGE_MIN minutes (default 60): a tab left open keeps polling the API,
  and that alone shouldn't keep the server up all night.
Activity is read from Caddy's JSON access log. Boot counts as activity, so a fresh start gets the full window.
"""
import json
import os
import subprocess
import time

LOG = "/var/log/caddy/access.log"
IDLE = float(os.environ.get("IDLE_MIN", 15)) * 60
PAGE = float(os.environ.get("PAGE_MIN", 60)) * 60
PAGES = ("/", "/exit", "/layout")                       # plus the shopper's /b/<token>


def booted_at():
    return time.time() - float(open("/proc/uptime").read().split()[0])


def last_activity():
    last_any = last_page = booted_at()
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2)
            start = max(0, f.tell() - 2_000_000)
            f.seek(start)
            lines = f.read().splitlines()
    except FileNotFoundError:
        return last_any, last_page
    if start:
        lines = lines[1:]                               # the first line is cut in half
    for raw in lines:
        try:
            e = json.loads(raw)
        except ValueError:
            continue
        ts, req = float(e.get("ts", 0)), e.get("request", {})
        uri = req.get("uri", "").split("?")[0]
        last_any = max(last_any, ts)
        if req.get("method") == "GET" and (uri in PAGES or uri.startswith("/b/")):
            last_page = max(last_page, ts)
    return last_any, last_page


if __name__ == "__main__":
    now = time.time()
    a, p = last_activity()
    if now - a > IDLE or now - p > PAGE:
        print(f"idle: last request {now - a:.0f} s ago, last page {now - p:.0f} s ago; stopping", flush=True)
        subprocess.run(["shutdown", "-h", "now"], check=False)
