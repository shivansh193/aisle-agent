"""Wake link for the Aisle Agent demo server.

Opening the function URL starts the stopped EC2 instance and shows a waiting page, which polls
?status=1 and forwards to the app once its /health answers. The instance stops itself when idle
(deploy/aws/idle_stop.py). Needs INSTANCE and APP_URL; the role may only start and describe instances.
"""
import json
import os
import urllib.request

import boto3

ec2 = boto3.client("ec2")
INSTANCE = os.environ["INSTANCE"]
APP = os.environ["APP_URL"].rstrip("/")


def app_ready():
    try:
        with urllib.request.urlopen(APP + "/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def respond(status, body, ctype, **headers):
    return {"statusCode": status, "body": body,
            "headers": {"content-type": ctype, "cache-control": "no-store", **headers}}


def lambda_handler(event, context):
    q = event.get("queryStringParameters") or {}
    inst = ec2.describe_instances(InstanceIds=[INSTANCE])["Reservations"][0]["Instances"][0]
    state = inst["State"]["Name"]
    if state == "stopped":                              # "stopping" waits for the next poll
        ec2.start_instances(InstanceIds=[INSTANCE])
        state = "pending"
    ready = state == "running" and app_ready()
    if "status" in q:
        return respond(200, json.dumps({"state": state, "ready": ready}), "application/json")
    if ready:
        return respond(302, "", "text/plain", location=APP + "/")
    return respond(200, PAGE.replace("__APP__", APP), "text/html; charset=utf-8")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aisle Agent · waking up</title>
<style>
  :root { --bg:#f6f3ec; --ink:#1d1d1b; --mute:#6b6a64; --line:#dcd7cb; --ok:#2f6f4f; }
  @media (prefers-color-scheme: dark) { :root { --bg:#161614; --ink:#ecebe6; --mute:#9a988f; --line:#33322e; --ok:#7cc29b; } }
  body { margin:0; min-height:100vh; display:grid; place-items:center; background:var(--bg); color:var(--ink);
         font:16px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width:30rem; padding:2rem 1rem; }
  h1 { font-size:1.6rem; margin:0 0 .4rem; letter-spacing:-.01em; }
  h1 b { color:var(--ok); }
  p { margin:.4rem 0; color:var(--mute); }
  .bar { height:4px; background:var(--line); border-radius:2px; overflow:hidden; margin:1.4rem 0 .8rem; }
  .bar i { display:block; height:100%; width:0; background:var(--ok); transition:width 1s linear; }
  code { font:13px ui-monospace, SFMono-Regular, Consolas, monospace; color:var(--ink); }
</style></head>
<body><main>
  <h1>Aisle Agent is <b>waking up</b></h1>
  <p>The demo server sleeps when nobody is using it, so it doesn't bill around the clock.
     It's starting now and usually takes about a minute.</p>
  <div class="bar"><i id="bar"></i></div>
  <p><code id="st">starting the server…</code></p>
  <p>You'll be taken to the app when it's ready. It goes back to sleep after 15 minutes without use.</p>
</main>
<script>
  const t0 = Date.now(), app = "__APP__";
  const label = { pending: "starting the server…", stopping: "finishing the last shutdown, then starting…",
                  stopped: "starting the server…", running: "server is up, starting the app…" };
  async function poll() {
    const s = Math.round((Date.now() - t0) / 1000);
    document.getElementById("bar").style.width = Math.min(95, s / 90 * 100) + "%";
    try {
      const r = await (await fetch("?status=1", { cache: "no-store" })).json();
      if (r.ready) { document.getElementById("bar").style.width = "100%"; location.replace(app + "/"); return; }
      document.getElementById("st").textContent = (label[r.state] || r.state) + " " + s + " s";
    } catch (e) { document.getElementById("st").textContent = "still waiting… " + s + " s"; }
    setTimeout(poll, 3000);
  }
  poll();
</script>
</body></html>
"""
