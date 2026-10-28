"""Record the demo walkthrough from the live system: a real browser drives the real UI (Playwright).

Runs on the server itself. Every number on screen comes from the live models (Nemotron via Token Factory).
  BASE=https://<host> AISLE_KEY=... python record_demo.py
Scenes (one recording each, in OUT):
  2_shelf.webm  the console with shelf footage fed in as the browser's camera (Chrome's fake-camera
                input; the console says "this device") -> draw the shelf zone -> takes print on the bill
                -> "not mine" on one line
  3_exit.webm   exit check -> the disputed line's evidence -> note -> remove -> total and record update
  4_layout.webm shop layout: the example shop -> shelves, stock, camera zones -> save (before the shelf scene)
  2_phone.webm  the shopper's phone (opened from the console's QR link) during the shelf scene -> "Not mine"
  5_stock.webm  the layout after the exit check: stock that followed the bill, the stock moves
marks.json      when things happened in each recording (seconds from its start), for the video edit
"""
import asyncio
import json
import os
import time

from playwright.async_api import async_playwright

BASE = os.environ["BASE"].rstrip("/")
KEY = os.environ["AISLE_KEY"]
OUT = os.environ.get("OUT", "/opt/ss/rec")
SHELF_Y4M = os.environ.get("SHELF_Y4M", "/opt/ss/rec/shelf.y4m")
W, H = 1440, 900
os.makedirs(OUT, exist_ok=True)
MARKS = {}
T0 = {}


def start_clock(scene):
    T0[scene] = time.monotonic()
    MARKS.setdefault(scene, {})


def mark(scene, name):
    MARKS[scene][name] = round(time.monotonic() - T0[scene], 2)
    print(scene, name, MARKS[scene][name], flush=True)


def save_marks():
    path = f"{OUT}/marks.json"
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(MARKS)
    json.dump(old, open(path, "w"), indent=1)

CURSOR = """
(() => {
  const mk = () => {
    if (document.getElementById('__cur')) return;
    const c = document.createElement('div'); c.id = '__cur';
    Object.assign(c.style, {position: 'fixed', width: '20px', height: '20px', borderRadius: '50%', zIndex: 2147483647,
      background: 'rgba(23,24,26,0.82)', border: '2.5px solid #F3F1EA', boxShadow: '0 2px 8px rgba(0,0,0,0.25)',
      pointerEvents: 'none', transform: 'translate(-50%,-50%)', transition: 'width .12s, height .12s, background .12s'});
    const p = JSON.parse(sessionStorage.getItem('__cur') || '[720,450]');
    c.style.left = p[0] + 'px'; c.style.top = p[1] + 'px';
    document.documentElement.appendChild(c);
    addEventListener('mousemove', (e) => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px';
      sessionStorage.setItem('__cur', JSON.stringify([e.clientX, e.clientY])); }, true);
    addEventListener('mousedown', () => { c.style.width = c.style.height = '30px'; c.style.background = 'rgba(29,107,66,0.85)'; }, true);
    addEventListener('mouseup', () => { c.style.width = c.style.height = '20px'; c.style.background = 'rgba(23,24,26,0.82)'; }, true);
  };
  if (document.readyState === 'loading') addEventListener('DOMContentLoaded', mk); else mk();
})();
"""

SHELF_CATALOGUE = [
    {"sku": "frozen", "name": "frozen food pack", "price": 220},
    {"sku": "tray", "name": "tray of sauce or ready meal", "price": 180},
    {"sku": "milk", "name": "carton of milk or juice", "price": 68},
    {"sku": "biscuits", "name": "packet of biscuits", "price": 40},
    {"sku": "cereal", "name": "box of cereal", "price": 145},
    {"sku": "jam", "name": "jar of jam", "price": 189},
]


async def glide(page, x, y, steps=28):
    await page.mouse.move(x, y, steps=steps)
    await page.wait_for_timeout(180)


async def click(page, selector, pause=500):
    el = page.locator(selector).first
    await el.scroll_into_view_if_needed()
    b = await el.bounding_box()
    await glide(page, b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
    await page.mouse.down()
    await page.wait_for_timeout(90)
    await page.mouse.up()
    await page.wait_for_timeout(pause)


async def type_into(page, selector, text, delay=55):
    await click(page, selector, pause=200)
    await page.keyboard.type(text, delay=delay)
    await page.wait_for_timeout(300)


async def drag_zone(page, box):
    """Drag a zone on the live feed, box in frame fractions."""
    await page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")         # the Draw button sits below the feed
    await page.wait_for_timeout(900)
    r = await page.evaluate("""() => { const o = document.getElementById('overlay').getBoundingClientRect(); const m = mediaRect();
        return {x: o.left + m.ox, y: o.top + m.oy, w: m.W, h: m.H}; }""")
    x0, y0 = r["x"] + box[0] * r["w"], r["y"] + box[1] * r["h"]
    x1, y1 = r["x"] + box[2] * r["w"], r["y"] + box[3] * r["h"]
    await glide(page, x0, y0)
    await page.mouse.down()
    await page.mouse.move(x1, y1, steps=40)
    await page.mouse.up()
    await page.wait_for_timeout(600)
    await page.locator("#zoneName").scroll_into_view_if_needed()


async def new_context(browser, name, storage, size=(W, H), mobile=False):
    ctx = await browser.new_context(viewport={"width": size[0], "height": size[1]}, record_video_dir=f"{OUT}/{name}",
                                    record_video_size={"width": size[0], "height": size[1]}, permissions=["camera"],
                                    is_mobile=mobile, has_touch=mobile)
    await ctx.add_init_script(CURSOR)
    await ctx.add_init_script("(() => { const s = %s; for (const k in s) localStorage.setItem(k, JSON.stringify(s[k])); })()"
                              % json.dumps(storage))
    return ctx


async def finish(ctx, name):
    page = ctx.pages[0]
    video = page.video
    await ctx.close()
    path = await video.path()
    os.replace(path, f"{OUT}/{name}.webm")
    print("recorded", name, flush=True)


async def scene_layout(pw):
    browser = await pw.chromium.launch()
    ctx = await new_context(browser, "4_layout", {"ss_key": KEY})
    await ctx.request.put(f"{BASE}/v1/layout", data={"name": "Corner shop", "products": [], "shelves": []},
                          headers={"Authorization": f"Bearer {KEY}"})                 # start from an empty shop
    page = await ctx.new_page()
    start_clock("layout")
    await page.goto(f"{BASE}/layout")
    await page.wait_for_timeout(2500)
    await click(page, "#example", pause=1800)
    mark("layout", "example")
    for i, pause in ((0, 3800), (3, 3200)):                                  # the freezer, then a low aisle
        await click(page, f'#plan g.s[data-i="{i}"] rect.b', pause=pause)
        mark("layout", f"shelf{i}")
    g = await page.locator('#plan g.s[data-i="2"] rect.b').bounding_box()     # move Aisle 1 a little
    cell = (await page.locator("#plan").bounding_box())["width"] / 40
    await glide(page, g["x"] + g["width"] / 2, g["y"] + g["height"] / 2)
    await page.mouse.down()
    await page.mouse.move(g["x"] + g["width"] / 2 + 3 * cell, g["y"] + g["height"] / 2 - cell, steps=30)
    await page.mouse.up()
    await page.wait_for_timeout(1500)
    await click(page, "#save", pause=2500)
    mark("layout", "saved")
    await glide(page, W * 0.86, H * 0.45, steps=40)                          # needs attention, stock moves
    await page.wait_for_timeout(3000)
    mark("layout", "end")
    await finish(ctx, "4_layout")
    await browser.close()


async def scene_shelf_and_exit(pw):
    browser = await pw.chromium.launch(args=[
        "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-video-capture={SHELF_Y4M}", "--autoplay-policy=no-user-gesture-required"])
    api = await browser.new_context()
    lay = await (await api.request.get(f"{BASE}/v1/layout/catalogue", headers={"Authorization": f"Bearer {KEY}"})).json()
    await api.close()
    ctx = await new_context(browser, "2_shelf", {"ss_source": "device", "ss_zones": [], "ss_key": KEY,
                                                 "ss_catalogue": lay["products"] or SHELF_CATALOGUE, "ss_photos": {},
                                                 "ss_layout_zones": lay["zones"]})
    page = await ctx.new_page()
    start_clock("shelf")
    await page.goto(f"{BASE}/")
    await page.wait_for_timeout(1500)
    await click(page, "#start", pause=5000)
    mark("shelf", "started")
    await click(page, "#draw", pause=500)
    await drag_zone(page, [0.01, 0.36, 0.54, 0.99])                        # the freezer shelf
    await page.fill("#zoneName", "")
    await type_into(page, "#zoneName", "Freezer")                          # named after the layout shelf: its products tick
    await page.wait_for_timeout(1200)
    await click(page, "#zoneSave", pause=1200)
    mark("shelf", "zone_saved")
    await page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
    await page.wait_for_timeout(800)
    url = await page.locator("#phoneLink").get_attribute("href")
    pctx = await new_context(browser, "2_phone", {}, size=(390, 844), mobile=True)
    phone = await pctx.new_page()
    mark("shelf", "phone_t0")                                              # the phone recording starts here
    start_clock("phone")
    await phone.goto(url)
    await glide(page, W * 0.82, H * 0.62, steps=40)                        # the QR under the bill
    await page.wait_for_timeout(22000)                                     # takes print on both screens
    mark("shelf", "takes_shown")
    if await phone.locator("[data-ask]").count():                          # the shopper: "Not mine"
        await click(phone, "[data-ask]", pause=1600)
        await click(phone, "[data-yes]", pause=1200)
        mark("shelf", "not_mine")
        mark("phone", "not_mine")
    await page.wait_for_timeout(5000)
    await page.mouse.wheel(0, 500)
    await page.wait_for_timeout(3000)
    await page.mouse.wheel(0, -500)
    await page.wait_for_timeout(2000)
    await page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
    await page.wait_for_timeout(600)
    await click(page, ".engbadge", pause=600)                              # the model trace: every Nemotron call
    mark("shelf", "trace")
    await glide(page, W * 0.86, H * 0.45, steps=30)
    await page.wait_for_timeout(6500)
    await click(page, "#traceX", pause=800)
    mark("shelf", "end")
    await finish(pctx, "2_phone")
    await finish(ctx, "2_shelf")

    ctx = await new_context(browser, "3_exit", {"ss_key": KEY})
    page = await ctx.new_page()
    start_clock("exit")
    await page.goto(f"{BASE}/exit")
    await page.wait_for_timeout(4000)                                      # opens on the shopper with most to check
    for i, (note, button) in enumerate([("Checked the bag: not taken", "#rm"), ("Seen in the basket", "#ap"),
                                        ("Seen in the basket", "#ap")]):
        if not await page.locator("#queue .q").count():
            break
        if i == 0:
            await click(page, "#queue .q", pause=2500)
        mark("exit", f"line{i}")
        try:                                                               # Nemotron-3-Super reads the evidence
            await page.wait_for_selector("#advice .advice .row", timeout=25000)
            mark("exit", f"advice{i}")
        except Exception:
            pass
        await glide(page, W * 0.55, H * 0.42, steps=36)                    # look over the before/after evidence
        await page.wait_for_timeout(3500)
        await page.mouse.wheel(0, 380)
        await page.wait_for_timeout(2500)                                  # what else it could have been
        await type_into(page, "#note", note, delay=45)
        await click(page, button, pause=3500)
        mark("exit", f"decided{i}")
        await page.mouse.wheel(0, -380)
        await page.wait_for_timeout(1200)
    if await page.locator("#others .q").count():
        await click(page, "#others .q", pause=4000)
    await glide(page, W * 0.86, H * 0.3, steps=40)                         # total and the record
    await page.wait_for_timeout(4000)
    mark("exit", "end")
    await finish(ctx, "3_exit")

    ctx = await new_context(browser, "5_stock", {"ss_key": KEY})
    page = await ctx.new_page()
    start_clock("stock")
    await page.goto(f"{BASE}/layout")
    await page.wait_for_timeout(2500)
    await click(page, '#plan g.s[data-i="0"] rect.b', pause=3500)           # the freezer after the takes
    await glide(page, W * 0.86, H * 0.7, steps=40)
    await page.mouse.wheel(0, 400)                                         # the stock moves
    await page.wait_for_timeout(4500)
    mark("stock", "end")
    await finish(ctx, "5_stock")
    await browser.close()


async def scene_agents(pw):
    """The shop assistant (Nemotron tool calls) and the end-of-day report (Nemotron-3-Ultra)."""
    browser = await pw.chromium.launch()
    ctx = await new_context(browser, "6_agents", {"ss_key": KEY})
    page = await ctx.new_page()
    start_clock("agents")
    await page.goto(f"{BASE}/")
    await page.wait_for_timeout(2000)
    await page.locator("#asForm").scroll_into_view_if_needed()
    await page.wait_for_timeout(800)
    for k, q in enumerate(["What's running low?", "Which shelf had the most uncertain takes, and why?"]):
        n = await page.locator("#asLog li.bot").count()
        await click(page, f'#asChips button:has-text("{q}")', pause=300)
        mark("agents", f"ask{k}")
        await page.wait_for_function(f"document.querySelectorAll('#asLog li.bot .meta').length > {n}", timeout=45000)
        mark("agents", f"answer{k}")
        await page.locator("#asLog li.bot").last.scroll_into_view_if_needed()
        await page.wait_for_timeout(5500)
    await page.goto(f"{BASE}/layout")
    await page.wait_for_timeout(1800)
    await click(page, "#dayReport", pause=300)
    mark("agents", "report_ask")
    await page.wait_for_selector("#reportMount .rep, #reportMount pre", timeout=60000)
    mark("agents", "report")
    await page.locator("#reportMount").scroll_into_view_if_needed()
    await glide(page, W * 0.85, H * 0.5, steps=30)
    await page.wait_for_timeout(7000)
    await page.mouse.wheel(0, 350)
    await page.wait_for_timeout(4000)
    mark("agents", "end")
    await finish(ctx, "6_agents")
    await browser.close()


async def main():
    async with async_playwright() as pw:
        which = os.environ.get("SCENES", "layout,shelf,agents").split(",")
        if "layout" in which:
            await scene_layout(pw)
        if "shelf" in which:
            await scene_shelf_and_exit(pw)
        if "agents" in which:
            await scene_agents(pw)
        save_marks()


if __name__ == "__main__":
    asyncio.run(main())
