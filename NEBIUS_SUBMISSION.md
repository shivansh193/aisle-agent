# Submission draft: Nebius × NVIDIA Global AI Hackathon

**Track:** Physical AI.

**Project name:** Aisle Agent

**Tagline (≤ 200 chars):** A shop's own camera becomes its checkout and stock clerk: NVIDIA Nemotron on Nebius Token
Factory decides what was taken, names it, advises at the exit and answers the owner.

## What it does

Small shops can't afford Amazon Go-style ceiling rigs, but almost all of them already have a camera
over the shelves. Aisle Agent turns that camera into a checkout and a stock clerk:

- **Takes go on the bill.** A hand goes into a shelf zone empty and comes out holding something:
  that's a take. Nemotron-3.5-Super-VL checks the hand just before and just after the visit, names
  the product among the ones stocked on that shelf, and a line goes onto the shopper's bill. Stock
  on that shelf goes down by one.
- **Put-backs are caught.** A hand that goes in holding something and comes out empty flags the line.
- **It never charges on a guess.** Uncertain lines are marked "verify at exit". If the model is
  unavailable, the visit is marked "unchecked" rather than guessed.
- **Exit check with an AI advisor.** Nemotron-3-Super reads each flagged line's evidence and explains
  it in plain English, recommending keep, remove or ask the shopper. A person decides. Both go in the
  audit log, and the page shows how often the advisor and staff agree.
- **The shopper's phone.** A QR code opens the shopper's own live bill, with a "Not mine" button.
- **Shelf watch and layout.** Between visits, the same camera reads how full each shelf is and
  raises restock alerts. A floor plan keeps stock per shelf.
- **Ask the shop.** The owner asks "what's running low?" or "why were the takes on Aisle 2 uncertain?"
  A Nemotron agent answers with tool calls over the live shop data.

## How Nemotron and Token Factory are used

| Model | Task | Why this model |
|---|---|---|
| `nvidia/Nemotron-3.5-Super-VL` | Hand holding a product (before/after each shelf visit), product naming among a shelf's products, shelf stock level, evidence captions | Vision, with logprobs on image prompts, so answers are calibrated probabilities instead of parsed text |
| `nvidia/nemotron-3-super-120b-a12b` | Exit advisor; shop assistant for multi-step "why" questions | Reasoning over structured evidence; tool calling |
| `nvidia/Nemotron-3_5-Lightning` | Shop assistant for quick lookups | Fast and cheap for single-tool answers |

**The design idea:** a funnel. A local pose model (YOLO11n-pose + ByteTrack) runs on every frame and
finds hands. Nemotron is called only when a hand actually visits a shelf, so hosted inference costs
**$0.17 per 1,000 shelf visits**. Decisions go through **vjev**, an open-source (Apache-2.0)
typed-decision layer that I extended with a Token Factory backend. It reads Nemotron's answer
logprobs, calibrates them into probabilities, abstains on timeouts, and records every call (model,
latency, tokens, cost) for the console's model trace.

**Deployment win:** with the VLM hosted, the server needs no GPU. Pose tracking runs at 27.6 fps
(320 px) or 17.4 fps (416 px) on 4 vCPU. Aisle Agent runs on a $0.18/hour CPU instance plus Token
Factory.

## Results (same items, hosted Nemotron vs a local 3B VLM on a T4)

| Check | Nemotron-3.5-Super-VL | Qwen2.5-VL-3B local |
|---|---|---|
| Naming within a shelf of 5, clean phone photos (300) | **95.3%** | 90.3% |
| Naming, CCTV-quality images | **78.7%** | 65.7% |
| Takes on 9 handheld stock clips (8 labelled) | **3 found**, 1 false | 2 found, 1 false |
| Shelf watch (484 real shelf crops) | AUROC 0.867, 83.3% at the alert threshold | AUROC 0.883 (different sample) |
| Latency per decision | 1.1–1.6 s p50 | ~0.3 s |

Take finding on close-up handheld stock footage is still the weak point. A fixed shop camera is the
target, and real-shop footage is the next test. The full evaluation through Token Factory cost **$0.12**.

## Feedback on Token Factory and the NVIDIA models

**What worked well**
1. **Logprobs on image prompts.** `logprobs` + `top_logprobs` work for Nemotron-3.5-Super-VL with an
   image in the prompt. That is what makes calibrated yes/no and choice decisions possible from a
   hosted VLM; it's the single most useful feature for us.
2. **OpenAI-compatible API.** A ~100-line stdlib client worked first time. Tool calling on
   `nemotron-3-super-120b-a12b` worked in the standard format (the assistant chose the right tool
   on its own).
3. **Price.** The whole evaluation (about 1,150 vision calls) cost $0.12. $0.17 per 1,000 shelf visits
   makes per-shop hosted inference viable.
4. **Accuracy.** Nemotron VL named products clearly better than a local 3B VLM, especially on
   low-quality CCTV-like images (+13 points).

**What needs work**
1. **Reasoning is on by default, and the switch is undocumented for VL.** With `max_tokens=1`, the
   first token was "The" (reasoning), and `content` was `null`. We tried four switches:
   `chat_template_kwargs={"enable_thinking": false}` and `reasoning_effort="none"` work; a
   `/no_think` system prompt and `thinking={"type":"disabled"}` are silently ignored. Please document
   the supported switch per model in the catalogue and quickstart. This cost us the most time.
2. **Model ID casing is inconsistent:** `nvidia/Nemotron-3.5-Super-VL` (dot),
   `nvidia/Nemotron-3_5-Lightning` (underscore), `nvidia/nemotron-3-super-120b-a12b` (lowercase).
   Easy to get wrong; aliases or one convention would help.
3. **Latency from India to us-north1:** p50 1.1–1.6 s, p95 1.7–2.5 s per single-token decision with
   one small image. Fine for once-per-visit checks, too slow for anything per frame. A region closer
   to South Asia, or a published latency table per region, would help.
4. **Image token cost is opaque.** A 64×64 test image was billed as 258 image tokens. Documenting the
   image token formula (size → tokens) would let us size crops for cost.
5. **Hallucination on empty images.** Asked to describe a hand in a plain black frame, Nemotron VL
   described "an open hand, palm up". Calibrated yes/no checks were more robust than free-text
   captions; a note on this in the VL model card would help.
6. **Calibration on degraded images:** expected calibration error rose from 0.03 (clean) to 0.12 (CCTV
   quality). A published calibration note, or per-model temperature guidance, would help builders who
   use logprobs.

**Nebius AI Cloud:** not used this time; the app runs on a CPU VM elsewhere and calls Token Factory.

## Links

- Repo: https://github.com/shivansh193/aisle-agent
- Video: (YouTube, under 3 min)
- Live demo: https://<ip-with-dashes>.sslip.io (shop key shared with judges in the testing notes)

No affiliation with or endorsement by Nebius, NVIDIA, Amazon or Ring.
