# Aisle Agent

**A shop's own camera becomes a checkout and a stock clerk.** Aisle Agent watches the shelves through
the camera a small shop already has. When a hand goes into a shelf empty and comes out holding
something, it puts that product on the shopper's bill and takes it off the shelf's stock. When it
isn't sure, it says so and leaves the line for a person at the exit, never charging on a guess.

The thinking is done by **NVIDIA Nemotron models through Nebius Token Factory**:

| Model (Token Factory) | What it does in Aisle Agent | When it runs |
|---|---|---|
| `nvidia/Nemotron-3.5-Super-VL` | Is this hand holding a product, just before and just after a shelf visit? How full is this shelf? Which product is this, among the ones stocked here? **Shelf audit:** is something on this shelf that doesn't belong, and what do its price labels say? | Only when a hand visits a shelf zone; per idle shelf every few seconds (stock) and every 5 minutes or on request (audit) |
| `nvidia/nemotron-3-super-120b-a12b` | **Exit advisor:** reads a flagged line's evidence and explains it to staff, with a keep / remove / ask-the-shopper recommendation. **Shop assistant** for multi-step questions. | When staff open a flagged line, or ask a "why" question |
| `nvidia/Nemotron-3_5-Lightning` | Shop assistant for quick lookups ("what's running low?") through tool calls over live shop data | On each question |
| `nvidia/Nemotron-3-Ultra-550b-a55b` | End-of-day report: sales, restock, uncertain takes and open audit findings | Once a day, or on request |

Cheap models stay local and run on every frame (YOLO11n-pose + ByteTrack find people and hands). A
hosted model is called only when something happens at a shelf. Calling an API on every frame would
be slow and expensive.

## How it works

```
camera (phone, laptop or IP camera over the shelf)
  └─ WebRTC (WHIP) · WebSocket JPEG frames ─▶ one session per camera, newest frame wins
       every frame, local:  YOLO11n-pose + ByteTrack ─▶ people, wrists, hand boxes
       a hand enters and leaves a shelf zone ─▶ visit
            Nemotron-3.5-Super-VL, calibrated by vjev:  P(holding) before vs after
               empty → holding  = take      ─▶ name it (Nemotron VL among that shelf's products,
                                                 or a local photo match on GPU boxes) ─▶ bill line, stock − 1
               holding → empty  = put-back  ─▶ flag the shopper's matching line
               model unavailable            ─▶ visit marked "unchecked" (never guessed)
       idle shelf, every 5 s:  Nemotron VL stock reading (empty / low / half / full) ─▶ restock alert
       idle shelf, every 5 min or on request:  shelf audit ─▶ wrong-shelf product, price label ≠ catalogue
  bill ─▶ shopper's phone (QR link, live, "Not mine")
       ─▶ exit check: evidence + Nemotron-3-Super advice; a person decides; both are logged
  shop layout ─▶ floor plan, stock per shelf, follows the bill
  shop assistant ─▶ Nemotron tool calls over sessions, events, stock, layout, exit decisions
```

**The rule that never changes:** the system only adds or flags bill lines. Only a person removes
one, and the audit log keeps the advisor's recommendation next to the human decision. The exit page
shows how often the advisor and staff agreed.

### Calibrated decisions from a hosted VLM: `vjev`

`vjev/` (Apache-2.0, vendored here) turns a vision-language model into **typed, calibrated
decisions** (yes/no, choice, score, box) instead of free text. The new
`vjev.tokenfactory.TokenFactoryBackend` runs it on Nemotron through Token Factory:

- **Answers come from logprobs, not parsed text.** One answer token with `top_logprobs`; the option
  tokens' log-probabilities are the logits, then calibrated (Platt / temperature).
- **Reasoning is switched off for decisions.** Nemotron reasons first by default, so the first token
  is "The…", not the answer. `chat_template_kwargs={"enable_thinking": false}` makes the answer the
  first token, with logprobs like Yes −0.02 / No −4.02.
- **Fallback:** if logprobs aren't available, it uses 5 samples at temperature 0.7 and the vote share.
- **Failures never become answers:** after retries with backoff, a timeout raises
  `BackendUnavailable`, and the visit stays unchecked or the line stays "verify at exit".
- **Concurrency, cost, trace:** a semaphore bounds concurrent requests, token cost is counted at
  catalogue prices, and every call (model, latency, tokens, $) shows in the console's model trace.

### Shelf audit

Between visits, Nemotron-3.5-Super-VL audits each shelf zone against the shop layout:

- **Wrong shelf:** "This shelf should hold only: … Is there clearly a product on it that is NOT one of those?"
  This is a calibrated yes/no from logprobs. If yes, a short follow-up names the product. On the live
  freezer footage it flagged "Greek yogurt" on the Freezer shelf (P 0.88) in 4.2 s.
- **Price labels:** the shelf crop is sent at a higher resolution, and Nemotron returns the labels it can
  read as JSON (`{"labels": [{"text", "price"}]}`), told never to guess a price. Each label is matched to
  a product on that shelf by its words and compared with the catalogue price the bill charges. A mismatch
  is a finding, because a shopper who trusts the label would be charged a different price.

Findings carry the shelf crop and go to the console, the layout plan and the assistant. Staff resolve
them: moved it back, it belongs here, label fixed, or charge the label price (which updates the
catalogue). **Not yet evaluated on a labelled set:** price reading depends on the labels being legible
to the camera. On our freezer footage the tags are too small and nothing is read, which is the correct
behaviour.

## Results

Hosted Nemotron vs a local VLM (Qwen2.5-VL-3B on an NVIDIA T4), scored on **the same items**
(`evals/`). Accuracy below is uncalibrated; the calibration step is reported separately.

| Check | Nemotron-3.5-Super-VL (Token Factory) | Qwen2.5-VL-3B (local T4) |
|---|---|---|
| Naming a product by name within a shelf of 5 (GroceryStoreDataset, 300 phone photos) | **95.3%** top-1, ECE 0.030 | 90.3%, ECE 0.031 |
| Same, CCTV-quality images (96 px, JPEG q30) | **78.7%**, ECE 0.120 | 65.7%, ECE 0.075 |
| Takes on 9 handheld stock clips, 8 hand-labelled takes (26 shelf visits) | **3 found**, 1 false | 2 found, 1 false |
| Shelf watch: empty gap vs stocked (484 real shelf crops) | AUROC 0.867; **83.3%** right at the alert threshold | AUROC **0.884**; 66.3% at the same threshold |
| Latency per decision | p50 1.1–1.6 s, p95 1.7–2.5 s | ~0.3 s |
| Cost | **$0.17 per 1,000 shelf visits**; $0.10 per 1,000 shelf readings | a GPU |

**Calibration.** Nemotron VL was overconfident on CCTV-quality images. One choice temperature
(T = 1.73), fitted on half the naming items and scored on the other half, cuts its expected
calibration error on CCTV images from **0.094 to 0.038** (clean: 0.045 → 0.050). The fitted file ships
as `aisle_agent/nemotron_calibration.json` and loads automatically, so "verify at exit" triggers on
honest confidence.

What this says:
- **Better at naming,** especially on CCTV-quality images (+13 points).
- **Slower per call.** That's fine for a check that runs once per shelf visit, and the frame loop never waits for it.
- **Overconfident on CCTV images until calibrated;** one fitted temperature fixes most of it.
- **Take finding is still the weak point** on close-up, handheld footage: 3 of 8. A fixed shop camera is the case it's built for, and footage from a real shop is the next test.

**No GPU needed.** With the vision-language model hosted, the only local model on every frame is
pose tracking. On a 4-vCPU box, YOLO11n-pose + ByteTrack runs at **27.6 fps at 320 px** and
17.4 fps at 416 px (`evals/cpu_pose`). A local SigLIP photo match takes 8.3 s per take on that CPU,
so on CPU boxes products are named by Nemotron VL instead (`naming="auto"`). Aisle Agent runs on a
cheap CPU server plus Token Factory.

## Run it

```bash
pip install -r requirements.txt
echo "TOKEN_FACTORY_API_KEY=<your Nebius Token Factory key>" > .env      # git-ignored
python -m aisle_agent.server --engine nemotron --port 8000               # http://127.0.0.1:8000
```

Then open the console at `/` on a phone or laptop and press Start, so the browser camera streams
over WebRTC. Draw a shelf zone on the feed and name it after a shelf from `/layout`. Take something
from that shelf.

| Page | What it is |
|---|---|
| `/` | Console: live feed with hands and zones, the bill, shopper QR code, **Shelf audit** findings, what it saw, **Ask the shop** assistant, **model trace** (every Nemotron call) |
| `/exit` | Exit check: lines to verify, before/after evidence, **Nemotron-3-Super advice**, keep/remove, agreement rate |
| `/layout` | Shop layout: floor plan, products, stock and minimums per shelf, each tied to a camera zone; audit findings flagged on the plan |
| `/b/<token>` | The shopper's phone: their bill, live, with "Not mine" |

Other engines: `--engine stub` (no models, scripted, for UI work) and `--engine vjev` (local
Qwen2.5-VL on a GPU). Set `AISLE_KEY` to require `Authorization: Bearer <key>` on the API.

Tests (no key and no network needed; the hosted calls are faked): `pytest tests` (61 tests).

### Deploy

`deploy/aws/` provisions an EC2 host (Caddy for HTTPS, systemd service) and deploys with
`ENGINE=nemotron HOST=<ip> bash deploy/aws/deploy.sh`. The Token Factory key goes from `.env` into a
root-only env file on the host.

### Evaluate

`evals/export` runs on Kaggle and exports the test items. `evals/eval/nemotron_eval.py` scores
Nemotron on them through Token Factory, using only HTTP calls locally, with a guard that stops if a
run is projected to cost more than $5. The whole evaluation above cost $0.12.

## API (selected)

| | |
|---|---|
| `POST /v1/sessions` | `{catalogue, zones?, options?}` → session with ingest, events and shopper links |
| `POST /v1/sessions/{id}/whip` · `WS …/ingest` · `POST …/frames` | video in |
| `WS /v1/sessions/{id}/events` | visit / take / put / unchecked / line / bill / stock / restock |
| `GET /v1/sessions/{id}/bill/{line}/evidence` · `…/advice` · `POST …/review` | exit check |
| `POST /v1/assistant` `{question, history?}` | shop assistant (answer, model used, tools used) |
| `GET /v1/models/trace` · `GET /v1/agents/stats` | every hosted call · advisor/staff agreement |
| `GET/PUT /v1/layout`, `POST /v1/layout/shelves/{id}/restock` | shop layout and stock |

## Credits and licence

Apache-2.0 (`LICENSE`). Evaluation data: GroceryStoreDataset (Klasson et al., WACV 2019), Retail
Shelf Void Detection (Kaggle), Pexels stock clips. No affiliation with or endorsement by Nebius,
NVIDIA or any camera maker.
