"""AI voiceover for the Aisle Agent demo, on Kaggle (CPU): Kokoro TTS (voice af_heart) laid over the rendered
video (aisle-agent-video output). The video's own music and SFX duck under the voice; -16 LUFS.

Each line starts at its section's time; a line that runs past the next one is sped up (at most 1.18x).
Outputs: aisle_agent_demo_voiceover.mp4, aisle_agent_demo_voiceover.srt (written forms, real timings),
         voiceover_fit.json
"""
import glob
import json
import os
import subprocess

import numpy as np

OUT, WORK = "/kaggle/working", "/kaggle/tmp/vo"
os.makedirs(WORK, exist_ok=True)
SR = 24000

# (start s, spoken text, written text for subtitles)
LINES = [
    (0.3, "Every corner shop already has a camera. Aisle Agent makes it the checkout.", None),
    (7.7, "Cheap pose tracking runs locally on every frame. NVIDIA Nemotron, through Nebius Token Factory, is called only when a hand visits a shelf.", None),
    (16.2, "First, the owner maps the shop: shelves, fridges and freezers, with stock, and the camera zone that watches each one.", None),
    (29.7, "Then the shelf. Any ordinary camera over the freezer aisle is enough.", None),
    (40.1, "The shopper scans a QR code, and watches their bill fill in on their phone.", None),
    (47.0, "Empty hand in, holding something out. Nemotron vision calls it a take, and names the product from that shelf's stock.", None),
    (56.5, "If it isn't sure, the line says, verify at exit. Nothing is ever charged on a guess.",
     "If it isn't sure, the line says \"verify at exit\". Nothing is ever charged on a guess."),
    (66.3, "Not theirs? One tap, and the line goes to staff.", None),
    (72.9, "Every Nemotron call is visible here, with its latency, tokens and cost.", None),
    (80.7, "Each answer comes from Nemotron's token probabilities, calibrated, with reasoning switched off. Fast, and honest about doubt.", None),
    (89.6, "At the exit, Nemotron Super reads the evidence for each flagged line, and recommends what to do.", None),
    (98.5, "It only advises. A person decides, and both go in the record.", None),
    (116.0, "Between visits, Nemotron vision audits each shelf against the layout, and reads its price labels.", None),
    (121.6, "Yogurt in the freezer doesn't belong there, so staff get a flag, with the shelf crop.", None),
    (128.0, "The owner can simply ask. Nemotron answers with tool calls over live shop data.", None),
    (137.4, "Quick lookups go to Lightning. Why questions go to Super.", "Quick lookups go to Lightning; \"why\" questions go to Super."),
    (143.8, "At closing, Nemotron Ultra writes the day's report: sales, restock, and where takes were uncertain.", None),
    (155.4, "On the same test items as a local vision model, Nemotron names products better, especially on CCTV quality images. Take finding is still the weak spot.",
     "On the same test items as a local vision model, Nemotron names products better, especially on CCTV-quality images. Take finding is still the weak spot."),
    (164.4, "And it runs on a CPU server, for about seventeen cents per thousand shelf visits.",
     "And it runs on a CPU server, for about $0.17 per 1,000 shelf visits."),
    (171.9, "Aisle Agent. The camera you already have, as your checkout.", None),
]


def sh(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr)[-2500:], flush=True)
    if r.returncode:
        raise SystemExit(f"failed: {cmd}")
    return r.stdout


video = glob.glob("/kaggle/input/**/aisle_agent_demo.mp4", recursive=True)[0]
DUR = float(sh(f'ffprobe -v error -show_entries format=duration -of csv=p=0 "{video}"').strip())
print("video", video, DUR, flush=True)

# ---- narration: Kokoro needs Python < 3.13; it runs in a 3.12 venv made by uv
sh("apt-get install -y -qq espeak-ng > /dev/null")
sh("pip install -q uv && uv venv -q -p 3.12 /kaggle/tmp/tts && uv pip install --python /kaggle/tmp/tts/bin/python "
   "'kokoro>=0.9.4' soundfile 'numpy<2.3' "
   "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl "
   "2>&1 | tail -2 && /kaggle/tmp/tts/bin/python -c 'import kokoro, numpy, soundfile, en_core_web_sm'")
json.dump([[t, s] for t, s, _ in LINES], open(f"{WORK}/lines.json", "w"))
open(f"{WORK}/tts.py", "w").write('''
import json, sys
import numpy as np, soundfile as sf
from kokoro import KPipeline
work = sys.argv[1]
tts = KPipeline(lang_code="a")
for i, (t, text) in enumerate(json.load(open(f"{work}/lines.json"))):
    audio = np.concatenate([np.asarray(a, dtype=np.float32) for _, _, a in tts(text, voice="af_heart", speed=1.0)])
    sf.write(f"{work}/l{i}.wav", audio, 24000)
    print(i, round(len(audio) / 24000, 2), flush=True)
''')
sh(f"/kaggle/tmp/tts/bin/python {WORK}/tts.py {WORK}")
import soundfile as sf  # noqa: E402

track = np.zeros(int(DUR * SR), dtype=np.float32)
report, cues = [], []
for i, (t, spoken, written) in enumerate(LINES):
    audio, _ = sf.read(f"{WORK}/l{i}.wav", dtype="float32")
    nxt = LINES[i + 1][0] if i + 1 < len(LINES) else DUR - 0.3
    room, need = nxt - t - 0.25, len(audio) / SR
    tempo = 1.0
    if need > room:
        tempo = min(1.18, need / room)
        sh(f"ffmpeg -loglevel error -y -i {WORK}/l{i}.wav -filter:a atempo={tempo:.3f} {WORK}/l{i}f.wav")
        audio, _ = sf.read(f"{WORK}/l{i}f.wav", dtype="float32")
    a0 = int(t * SR)
    seg = audio[: len(track) - a0]
    track[a0:a0 + len(seg)] += seg
    secs = len(audio) / SR
    report.append({"t": t, "seconds": round(secs, 2), "room": round(room, 2), "tempo": round(tempo, 3),
                   "fits": secs <= room + 0.05, "text": spoken})
    cues.append((t, t + secs, written or spoken))
    print(report[-1], flush=True)
peak = float(np.abs(track).max()) or 1.0
sf.write(f"{WORK}/voice.wav", track * (0.89 / peak), SR)
json.dump(report, open(f"{OUT}/voiceover_fit.json", "w"), indent=1)

# ---- mix: the video's music + SFX duck under the voice
sh(f'ffmpeg -loglevel error -y -i "{video}" -i {WORK}/voice.wav -filter_complex '
   f'"[1:a]aresample=48000,asplit=2[v][vk];[0:a]aresample=48000[bg];[bg][vk]sidechaincompress=threshold=0.02:ratio=8:attack=15:release=350[bd];'
   f'[bd][v]amix=inputs=2:duration=first:normalize=0:weights=1 1.7,loudnorm=I=-16:TP=-1.5:LRA=11[a]" '
   f'-map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k -ar 48000 -movflags +faststart {OUT}/aisle_agent_demo_voiceover.mp4')


# ---- subtitles from the placed timings (written forms), split for reading
def ts(x):
    ms = int(round(x * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def split(text, limit=84):
    if len(text) <= limit:
        return [text]
    for marks in (".?:", ","):
        cuts = [i + 1 for i, c in enumerate(text) if c in marks and 10 < i < len(text) - 10]
        if cuts:
            i = min(cuts, key=lambda c: abs(c - len(text) / 2))
            return split(text[:i].strip(), limit) + split(text[i:].strip(), limit)
    return [text]


def wrap(t, w=42):
    if len(t) <= w:
        return t
    sp = [i for i, c in enumerate(t) if c == " "]
    i = min(sp, key=lambda k: abs(k - len(t) / 2))
    return t[:i] + "\n" + t[i + 1:]


out = []
for a, b, text in cues:
    parts = split(text)
    total, t = sum(len(p) for p in parts), a
    for p in parts:
        d = (b - a) * len(p) / total
        out.append([t, t + d, wrap(p)])
        t += d
for k in range(len(out)):
    nxt = out[k + 1][0] - 0.04 if k + 1 < len(out) else DUR
    out[k][1] = min(out[k][1] + 0.25, nxt)
open(f"{OUT}/aisle_agent_demo_voiceover.srt", "w", encoding="utf-8").write(
    "\n".join(f"{n}\n{ts(a)} --> {ts(b)}\n{t}\n" for n, (a, b, t) in enumerate(out, 1)))
sh(f"ffprobe -v error -show_entries format=duration:stream=codec_name -of compact {OUT}/aisle_agent_demo_voiceover.mp4")
sh(f"ffmpeg -hide_banner -i {OUT}/aisle_agent_demo_voiceover.mp4 -af volumedetect -vn -f null - 2>&1 | grep -E 'mean_volume|max_volume' || true")
print(json.dumps({"lines": len(report), "all_fit": all(r["fits"] for r in report), "cues": len(out)}))
