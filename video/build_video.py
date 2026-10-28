"""Hyperframes composition for the Aisle Agent demo (1920x1080, 30 fps, ~2:52), built around screen
recordings of the live system (deploy/record/record_demo.py on the CPU server, real Nemotron calls):

  4_layout.mp4  shop layout        2_shelf.mp4 + 2_phone.mp4  takes on the bill, shopper's phone, model trace
  3_exit.mp4    exit advisor        6_agents.mp4               shop assistant, end-of-day report

    python build_video.py --clips <dir with the mp4s and marks.json> --assets <dir with sfx/, fonts/, gsap.min.js> --out composition/

Cuts and captions are placed from marks.json, so the edit follows the real run. Captions sit in a
strip under the recording. Every number on the slides is measured (see README / evals/).
"""
import argparse
import html
import json
import os
import shutil

esc = html.escape


def flow(items):
    out = []
    for i, (h, t) in enumerate(items):
        out.append(f'<div class="box it"><b>{esc(h)}</b><span>{t}</span></div>')
        if i < len(items) - 1:
            out.append('<div class="arr it">→</div>')
    return '<div class="flow">' + "".join(out) + "</div>"


def grid(cells, cols=3):
    return (f'<div class="grid" style="grid-template-columns:repeat({cols}, 1fr)">' +
            "".join(f'<div class="cell it"><div class="ch">{esc(h)}</div><div class="cb">{b}</div></div>' for h, b in cells) +
            "</div>")


def table(rows):
    head = '<div class="tr th it"><span>Same items</span><span>Nemotron-3.5-Super-VL<br><small>Token Factory</small></span><span>Qwen2.5-VL-3B<br><small>local T4</small></span></div>'
    return '<div class="tbl">' + head + "".join(
        f'<div class="tr it"><span>{a}</span><span class="nv">{b}</span><span>{c}</span></div>' for a, b, c in rows) + "</div>"


def plan(M):
    segs, caps, cards, chapters, slides = [], [], [], [], []
    t = 0.0

    def seg(clip, m0, m1, kind="screen", g=None, advance=True):
        nonlocal t
        g0 = t if g is None else g
        segs.append({"id": f"v{len(segs)}", "clip": clip, "m0": round(m0, 2), "g0": round(g0, 2), "d": round(m1 - m0, 2), "kind": kind})
        if advance:
            t = g0 + (m1 - m0)
        return g0

    def card(n, title, d=1.5):
        nonlocal t
        cards.append((round(t, 2), d, n, title))
        t += d

    def slide(sid, d, kicker, title, body):
        nonlocal t
        slides.append((sid, round(t, 2), d, kicker, title, body))
        t += d

    def cap(g, text):
        caps.append((round(g, 2), text))

    t = 7.5
    slide("sA", 8.5, "At a glance", "A shop camera with a checkout and a stock clerk behind it", flow([
        ("Camera", "Phone, laptop or IP camera over the shelf"),
        ("Every frame · local", "YOLO11n-pose + ByteTrack find hands · 17 fps on 4 vCPU"),
        ("A shelf visit · hosted", "<b class='g'>Nemotron-3.5-Super-VL</b>: holding before / after? Which product? How full?"),
        ("State", "Bill, evidence, stock per shelf, shopper's phone"),
        ("Agents · hosted", "<b class='g'>Nemotron-3-Super</b> exit advisor and assistant · Lightning lookups · Ultra day report"),
    ]) + '<div class="foot it">All NVIDIA models via Nebius Token Factory · recorded live on a CPU-only server · every call shown in the model trace</div>')

    L = M["layout"]
    card("1", "Map the shop")
    c0 = t
    m0 = max(0.5, L["example"] - 1.0)
    g = seg("4_layout.mp4", m0, m0 + 13.0)
    cap(g + 0.2, "The owner maps the shop: shelves, fridges and freezers, with stock and minimums.")
    cap(g + L["shelf3"] - m0 - 3.4, "Aisle 2 is already low. Each shelf is tied to the camera zone that watches it.")
    chapters.append((c0, t, "1 · Shop layout"))

    S, P = M["shelf"], M["phone"]
    card("2", "A take goes on the bill")
    c0 = t
    g = seg("2_shelf.mp4", S["zone_saved"] - 8.5, S["zone_saved"] + 0.3, "left")
    cap(g + 0.2, "The freezer aisle through an ordinary camera. The zone is named after its shelf.")
    gb = t
    b1 = S["phone_t0"] + P["not_mine"] + 3.5
    seg("2_shelf.mp4", S["phone_t0"], b1, "left")
    seg("2_phone.mp4", 0.0, b1 - S["phone_t0"], "phone", g=gb, advance=False)
    phone_from = gb
    cap(gb + 0.3, "The shopper scans a QR code and follows their bill live.")
    cap(gb + 7.0, "Empty hand in, holding something out: Nemotron VL calls it a take and names the product.")
    cap(gb + 16.0, "Anything uncertain says \"verify at exit\". Nothing is charged on a guess.")
    cap(gb + P["not_mine"] - 3.0, "Not theirs? The shopper taps \"Not mine\". The line goes to staff.")
    g = seg("2_shelf.mp4", S["trace"] - 0.4, S["trace"] + 7.4)
    cap(g + 0.2, "The model trace: every Nemotron call, with its latency, tokens and cost.")
    chapters.append((c0, t, "2 · Bill at the shelf"))
    shelf_chapter = chapters[-1]

    slide("sC", 10.0, "How a take is decided", "Nemotron only runs when a hand visits a shelf", grid([
        ("① Every frame · local", "Pose tracking finds wrists; a hand box sits just past the wrist. No API call per frame."),
        ("② A shelf visit", "The hand just before it entered and just after it left → <b class='g'>Nemotron-3.5-Super-VL</b>: holding a product?"),
        ("③ Logprobs, not text", "One answer token with <code>top_logprobs</code> → P(yes). Reasoning switched off so the answer is the first token."),
        ("④ Calibrated (vjev)", "Temperature fitted on held-out items: CCTV calibration error <em>0.094 → 0.038</em>."),
        ("⑤ Name it", "Nemotron VL picks among that shelf's products: <em>95.3%</em> right in a shelf of 5."),
        ("⑥ Never a guess", "Below 0.5 → \"verify at exit\". Model unavailable → visit marked \"unchecked\"."),
    ]))

    E = M["exit"]
    card("3", "Check it at the exit")
    c0 = t
    e0 = E["line0"] - 1.0
    g = seg("3_exit.mp4", e0, E["decided0"] + 1.0)
    cap(g + 0.3, "At the counter, Nemotron-3-Super reads each flagged line's evidence…")
    cap(g + 6.5, "…and recommends keep, remove or ask the shopper. A person decides.")
    g = seg("3_exit.mp4", E["end"] - 7.5, E["end"])
    cap(g + 0.2, "Advice and decision both go in the record, with the agreement rate.")
    chapters.append((c0, t, "3 · Exit check"))

    A = M["agents"]
    card("4", "Ask the shop")
    c0 = t
    a0 = max(0.5, A["ask0"] - 1.6)
    g = seg("6_agents.mp4", a0, A["answer1"] + 6.0)
    cap(g + 0.2, "The owner asks. Nemotron answers with tool calls over live shop data.")
    cap(g + A["ask1"] - a0, "Lightning for quick lookups; Super for \"why\" questions.")
    r0 = A["report_ask"] - 0.8
    g = seg("6_agents.mp4", r0, min(A["end"], A["report"] + 10.0))
    cap(g + 0.2, "End of day: Nemotron-3-Ultra writes the report: sales, restock, uncertain takes.")
    chapters.append((c0, t, "4 · Ask the shop"))

    slide("sD", 10.0, "Results · measured, same items", "Hosted Nemotron vs a local 3B VLM", table([
        ("Naming, shelf of 5 · clean photos", "<em>95.3%</em>", "90.3%"),
        ("Naming · CCTV-quality images", "<em>78.7%</em>", "65.7%"),
        ("Shelf watch · right at the alert threshold", "<em>83.3%</em>", "66.3%"),
        ("Takes on handheld stock clips (8 labelled)", "3 found · <w>still the weak point</w>", "2 found"),
        ("Latency per decision", "1.1–1.6 s", "0.3 s"),
    ]) + '<div class="foot it">GroceryStoreDataset · Retail Shelf Void Detection · Pexels clips · scripts and numbers in the repo</div>')
    slide("sE", 9.0, "No GPU, small bill", "A CPU server plus Token Factory", grid([
        ("Server", "<em>c7i.xlarge</em>, 4 vCPU, no GPU · about $0.18 an hour"),
        ("Every frame", "Pose tracking at <em>17 fps</em> (416 px) on CPU"),
        ("Per shop", "<em>$0.17</em> of Nemotron per 1,000 shelf visits"),
        ("Whole evaluation", "About 1,150 vision calls for <em>$0.12</em>"),
        ("Agents", "Day report on Nemotron-3-Ultra in about 1 s"),
        ("Open source", "vjev's Token Factory backend: calibrated decisions from a hosted VLM, Apache-2.0"),
    ]))
    outro = t
    return {"segs": segs, "caps": caps, "cards": cards, "chapters": chapters, "slides": slides,
            "phone_from": phone_from, "shelf_chapter": shelf_chapter, "outro": round(outro, 2),
            "duration": round(outro + 5.5, 2)}


def build(clips, assets, out, marks):
    P = plan(marks)
    D = P["duration"]
    os.makedirs(f"{out}/assets", exist_ok=True)
    for c in sorted({s["clip"] for s in P["segs"]}):
        shutil.copy(f"{clips}/{c}", f"{out}/assets/{c}")
    for sub in ("sfx", "fonts"):
        if os.path.isdir(f"{assets}/{sub}"):
            shutil.copytree(f"{assets}/{sub}", f"{out}/assets/{sub}", dirs_exist_ok=True)
    shutil.copy(f"{assets}/gsap.min.js", f"{out}/assets/gsap.min.js")
    have = set(os.listdir(f"{out}/assets/fonts")) if os.path.isdir(f"{out}/assets/fonts") else set()
    faces = "".join(f"@font-face {{ font-family:'{fam}'; src:url('assets/fonts/{f}'); font-weight:100 900; }}\n"
                    for fam, f in (("Geist", "Geist.woff2"), ("Geist Mono", "GeistMono.woff2")) if f in have)
    sh = P["shelf_chapter"]

    videos = "\n".join(
        f'  <video id="{s["id"]}" class="{s["kind"]}" src="assets/{s["clip"]}" muted playsinline data-start="{s["g0"]}" '
        f'data-duration="{s["d"]}" data-media-start="{s["m0"]}" data-track-index="{4 if s["kind"] == "phone" else 2 + n % 2}"></video>'
        for n, s in enumerate(P["segs"]))
    strips = "\n".join(
        f'  <div id="ch{n}" class="strip" data-start="{a}" data-duration="{round(b - a, 2)}" data-track-index="6">'
        f'<span class="tag">{esc(t)}</span><span class="cap" id="cap{n}"></span></div>' for n, (a, b, t) in enumerate(P["chapters"]))
    cards = "\n".join(
        f'  <section id="card{n}" class="clip center" data-start="{s}" data-duration="{d}" data-track-index="5">'
        f'<div class="cnum">{esc(k)}</div><div class="ctitle">{esc(t)}</div></section>' for n, (s, d, k, t) in enumerate(P["cards"]))
    slides = "\n".join(
        f'  <section id="{i}" class="clip slide" data-start="{s}" data-duration="{d}" data-track-index="1">'
        f'<div class="kicker">{esc(k)}</div><div class="stitle">{esc(t)}</div>{body}</section>' for i, s, d, k, t, body in P["slides"])
    phone_end = next(s for s in P["segs"] if s["clip"] == "2_phone.mp4")
    holder = (f'  <div id="ph" class="phone holder" data-start="{sh[0]}" data-duration="{round(P["phone_from"] - sh[0] + 0.4, 2)}" '
              f'data-track-index="7"><div><b>The shopper\'s phone</b><span>opens from the QR code on the console</span></div></div>')

    js = []
    for n, (a, b, _) in enumerate(P["chapters"]):
        for s, t in [(s, t) for s, t in P["caps"] if a <= s < b]:
            js.append(f'  tl.call(() => setCap("cap{n}", {json.dumps(t)}), null, {s});')
            js.append(f'  tl.fromTo("#cap{n}", {{opacity:0, y:6}}, {{opacity:1, y:0, duration:0.25, ease}}, {s});')
    for s in P["segs"]:
        js.append(f'  tl.fromTo("#{s["id"]}", {{opacity:0}}, {{opacity:1, duration:0.25}}, {s["g0"]});')
        js.append(f'  tl.to("#{s["id"]}", {{opacity:0, duration:0.2}}, {round(s["g0"] + s["d"] - 0.2, 2)});')
    js.append(f'  tl.fromTo("#ph", {{opacity:0}}, {{opacity:1, duration:0.3}}, {sh[0]});')
    for n, (s, d, _, _) in enumerate(P["cards"]):
        js.append(f'  tl.fromTo("#card{n} .ctitle", {{opacity:0, y:10}}, {{opacity:1, y:0, duration:0.35, ease}}, {s + 0.05});')
        js.append(f'  tl.to("#card{n}", {{opacity:0, duration:0.2}}, {round(s + d - 0.2, 2)});')
    for i, s, d, *_ in P["slides"]:
        js.append(f'  tl.fromTo("#{i} .kicker, #{i} .stitle", {{opacity:0, y:10}}, {{opacity:1, y:0, duration:0.35, ease, stagger:0.12}}, {s + 0.05});')
        js.append(f'  tl.fromTo("#{i} .it", {{opacity:0, y:12}}, {{opacity:1, y:0, duration:0.35, ease, stagger:0.16}}, {s + 0.4});')
        js.append(f'  tl.to("#{i}", {{opacity:0, duration:0.25}}, {round(s + d - 0.25, 2)});')
    js = "\n".join(js)
    sfx = [4.0] + [c[0] for c in P["cards"]] + [P["outro"] + 0.1]
    audio = "\n".join(
        f'  <audio id="sx{i}" src="assets/sfx/{"impact/impactSoft_medium_001" if i in (0, len(sfx) - 1) else "interface/drop_001"}.ogg" '
        f'data-start="{round(s, 2)}" data-duration="0.6" data-volume="0.4" data-track-index="{11 + i}"></audio>' for i, s in enumerate(sfx))
    O = P["outro"]

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<script src="assets/gsap.min.js"></script>
<style>
  {faces}
  :root {{ --ground:#F3F1EA; --paper:#FBFAF6; --card:#FFFFFF; --ink:#17181A; --ink2:#55585E; --rule:#DEDAD0; --green:#1D6B42; --amber:#A85A00; --nv:#76B900; }}
  html, body {{ margin:0; background:var(--ground); }}
  #root {{ position:relative; width:1920px; height:1080px; overflow:hidden; background:var(--ground);
          font-family:'Geist', 'Inter', 'DejaVu Sans', system-ui, sans-serif; color:var(--ink); }}
  .clip {{ position:absolute; inset:0; background:var(--ground); }}
  .center {{ display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; }}
  .screen, .left, .phone {{ position:absolute; object-fit:cover; background:var(--paper); box-shadow:0 0 0 1px var(--rule), 0 18px 50px rgba(23,24,26,0.14); }}
  .screen {{ left:192px; top:18px; width:1536px; height:960px; border-radius:14px; }}
  .left {{ left:40px; top:40px; width:1392px; height:870px; border-radius:14px; }}
  .phone {{ left:1472px; top:40px; width:402px; height:870px; border-radius:34px; box-shadow:0 0 0 7px var(--ink), 0 18px 50px rgba(23,24,26,0.2); }}
  .holder {{ display:flex; align-items:center; justify-content:center; text-align:center; }}
  .holder b {{ display:block; font-size:28px; }} .holder span {{ display:block; margin-top:10px; font-size:20px; color:var(--ink2); padding:0 40px; }}
  .strip {{ position:absolute; left:192px; width:1536px; top:992px; height:76px; display:flex; align-items:center; gap:26px; }}
  .tag {{ flex:none; font-family:'Geist Mono', 'DejaVu Sans Mono', monospace; font-size:20px; color:var(--paper); background:var(--ink); border-radius:8px; padding:8px 14px; }}
  .cap {{ font-size:30px; font-weight:500; letter-spacing:-0.2px; }}
  .h1 {{ font-size:72px; font-weight:600; letter-spacing:-1.8px; }}
  .h1b {{ font-size:72px; font-weight:600; letter-spacing:-1.8px; color:var(--green); margin-top:10px; }}
  .title {{ font-size:136px; font-weight:700; letter-spacing:-4px; }}
  .sub {{ font-size:38px; color:var(--ink2); margin-top:14px; }}
  .meta {{ position:absolute; bottom:64px; font-family:'Geist Mono', 'DejaVu Sans Mono', monospace; font-size:21px; color:var(--ink2); }}
  .meta .dot {{ display:inline-block; width:12px; height:12px; border-radius:3px; background:var(--nv); margin-right:10px; vertical-align:0; }}
  .cnum {{ font-family:'Geist Mono', monospace; font-size:28px; color:var(--paper); background:var(--green); width:58px; height:58px; border-radius:50%;
          display:flex; align-items:center; justify-content:center; }}
  .ctitle {{ font-size:78px; font-weight:600; letter-spacing:-2px; margin-top:22px; }}
  .slide {{ padding:84px 120px 0; box-sizing:border-box; }}
  .kicker {{ font-family:'Geist Mono', 'DejaVu Sans Mono', monospace; font-size:22px; color:var(--green); letter-spacing:1.5px; text-transform:uppercase; }}
  .stitle {{ font-size:52px; font-weight:650; letter-spacing:-1.4px; margin:12px 0 44px; }}
  .grid {{ display:grid; gap:22px; }}
  .cell {{ background:var(--card); border:1.5px solid var(--rule); border-radius:14px; padding:24px 26px; min-height:230px; box-sizing:border-box; }}
  .ch {{ font-family:'Geist Mono', 'DejaVu Sans Mono', monospace; font-size:21px; font-weight:600; color:var(--green); margin-bottom:12px; }}
  .cb {{ font-size:25px; line-height:1.42; }}
  .cb code {{ font-family:'Geist Mono', monospace; font-size:21px; background:#EFECE3; padding:1px 6px; border-radius:5px; }}
  em {{ font-style:normal; font-family:'Geist Mono', monospace; font-weight:700; color:var(--green); }}
  w {{ color:var(--amber); font-weight:600; }}
  .g {{ color:var(--green); }}
  .flow {{ display:flex; align-items:stretch; gap:12px; margin-top:30px; }}
  .box {{ flex:1; background:var(--card); border:1.5px solid var(--rule); border-radius:14px; padding:26px 22px; }}
  .box b {{ display:block; font-size:27px; margin-bottom:12px; }} .box span {{ font-size:23px; line-height:1.42; color:var(--ink2); }}
  .arr {{ display:flex; align-items:center; font-size:40px; color:var(--green); }}
  .foot {{ margin-top:48px; font-family:'Geist Mono', 'DejaVu Sans Mono', monospace; font-size:22px; color:var(--ink2); }}
  .tbl {{ background:var(--card); border:1.5px solid var(--rule); border-radius:14px; overflow:hidden; }}
  .tr {{ display:grid; grid-template-columns:1.6fr 1fr 1fr; padding:22px 30px; border-top:1.5px solid var(--rule); font-size:29px; align-items:center; }}
  .tr.th {{ border-top:0; background:var(--paper); font-family:'Geist Mono', monospace; font-size:21px; color:var(--ink2); }}
  .tr.th small {{ font-size:17px; }} .tr .nv {{ font-weight:600; }}
  .footer {{ position:absolute; bottom:44px; width:100%; text-align:center; font-size:19px; color:var(--ink2); line-height:1.6; }}
</style>
</head>
<body>
<div id="root" data-composition-id="main" data-start="0" data-duration="{D}" data-width="1920" data-height="1080" data-fps="30">

  <section id="hook" class="clip center" data-start="0" data-duration="4" data-track-index="1">
    <div id="h1" class="h1">Every corner shop already has a camera.</div>
    <div id="h2" class="h1b">Aisle Agent makes it the checkout.</div>
  </section>

  <section id="title" class="clip center" data-start="4" data-duration="3.5" data-track-index="1">
    <div id="t1" class="title">Aisle Agent</div>
    <div id="t2" class="sub">A shop's own camera as its checkout and stock clerk.</div>
    <div id="t3" class="meta"><span class="dot"></span>NVIDIA Nemotron on Nebius Token Factory · recorded live</div>
  </section>

{slides}

{cards}

{holder}

{videos}

{strips}

  <section id="outro" class="clip center" data-start="{O}" data-duration="5.5" data-track-index="1">
    <div id="o1" class="title">Aisle Agent</div>
    <div id="o2" class="sub">github.com/shivansh193/aisle-agent</div>
    <div class="footer">Demo catalogue and prices · shelf footage: Aamir Somewhere (Pexels) ·
      no affiliation with or endorsement by Nebius or NVIDIA</div>
  </section>

{audio}
</div>

<script>
  window.__timelines = window.__timelines || {{}};
  const tl = gsap.timeline({{ paused: true }});
  const ease = "power2.out";
  const setCap = (id, t) => {{ document.getElementById(id).textContent = t; }};
  tl.fromTo("#h1", {{opacity:0, y:16}}, {{opacity:1, y:0, duration:0.5, ease}}, 0.15);
  tl.fromTo("#h2", {{opacity:0, y:16}}, {{opacity:1, y:0, duration:0.5, ease}}, 1.2);
  tl.to("#hook", {{opacity:0, duration:0.25}}, 3.75);
  tl.fromTo("#t1", {{opacity:0, scale:0.97}}, {{opacity:1, scale:1, duration:0.6, ease}}, 4.05);
  tl.fromTo("#t2", {{opacity:0}}, {{opacity:1, duration:0.4}}, 4.5);
  tl.fromTo("#t3", {{opacity:0}}, {{opacity:1, duration:0.4}}, 4.9);
  tl.to("#title", {{opacity:0, duration:0.25}}, 7.25);
{js}
  tl.fromTo("#o1", {{opacity:0, scale:0.97}}, {{opacity:1, scale:1, duration:0.6, ease}}, {O} + 0.1);
  tl.fromTo("#o2", {{opacity:0}}, {{opacity:1, duration:0.5}}, {O} + 0.6);
  window.__timelines.main = tl;
</script>
</body>
</html>
"""
    open(f"{out}/index.html", "w", encoding="utf-8").write(doc)
    shelf = next(s for s in P["segs"] if s["kind"] == "phone")
    timing = dict(P, poster_t=round(shelf["g0"] + 18.0, 2))
    json.dump(timing, open(f"{out}/timing.json", "w"), indent=1, default=str)
    return timing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("clips", "assets", "out"):
        ap.add_argument(f"--{a}", required=True)
    a = ap.parse_args()
    t = build(a.clips, a.assets, a.out, json.load(open(f"{a.clips}/marks.json")))
    print(json.dumps({"duration": t["duration"], "poster_t": t["poster_t"], "chapters": t["chapters"]}))
