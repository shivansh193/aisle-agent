"""Render the Aisle Agent demo video with Hyperframes on Kaggle (CPU).

Inputs:  aisle-agent-recordings (screen recordings of the live system + marks.json + build_video.py),
         aisle-agent-video-assets (interface SFX and the music bed)
Steps:   Node 22 + FFmpeg + Geist + GSAP -> composition -> hyperframes lint + check -> render
         -> poster frame baked as frame 0 -> music bed mixed under the SFX (-16 LUFS) -> preview sheet
Outputs: aisle_agent_demo.mp4, poster.jpg, preview.jpg, composition.zip, lint.txt, check.txt, render.txt
"""
import glob
import json
import os
import shutil
import subprocess
import zipfile

OUT = "/kaggle/working"
WORK = "/kaggle/tmp/brag"
os.makedirs(WORK, exist_ok=True)
ENV = dict(os.environ)


def sh(cmd, log=None, check=True, cwd=None, timeout=3600):
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=ENV, capture_output=True, text=True, timeout=timeout)
    text = (r.stdout or "") + (r.stderr or "")
    print(text[-4000:], flush=True)
    if log:
        open(f"{OUT}/{log}", "a").write(f"$ {cmd}\n{text}\n[exit {r.returncode}]\n")
    if check and r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {cmd}")
    return r.returncode, text


def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    return hits[0] if hits else None


# ---- inputs
assets = f"{WORK}/assets"
os.makedirs(assets, exist_ok=True)
for sub in ("music", "sfx"):
    src = find(sub)
    if src and os.path.isdir(src):
        shutil.copytree(src, f"{assets}/{sub}", dirs_exist_ok=True)
for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
    zipfile.ZipFile(z).extractall(f"{WORK}/unz/{os.path.splitext(os.path.basename(z))[0]}")
    for sub in ("sfx",):
        for d in glob.glob(f"{WORK}/unz/**/{sub}", recursive=True):
            shutil.copytree(d, f"{assets}/{sub}", dirs_exist_ok=True)
nested = f"{assets}/sfx/sfx"                                          # zipped uploads can nest one level
if os.path.isdir(nested):
    shutil.copytree(nested, f"{assets}/sfx", dirs_exist_ok=True)
builder = find("build_video.py")
clips = os.path.dirname(find("2_shelf.mp4"))
print("builder", builder, "clips", clips, flush=True)
print(sh(f"find {assets} -type f | head -8")[1])

# ---- toolchain
node_dir = f"{WORK}/node"
if not os.path.exists(f"{node_dir}/bin/node"):
    sh(f"curl -sL https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.xz -o {WORK}/node.tar.xz")
    sh(f"mkdir -p {node_dir} && tar -xJf {WORK}/node.tar.xz -C {node_dir} --strip-components=1")
ENV["PATH"] = f"{node_dir}/bin:" + ENV["PATH"]
if shutil.which("ffmpeg", path=ENV["PATH"]) is None:
    sh(f"curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o {WORK}/ff.tar.xz")
    sh(f"mkdir -p {WORK}/ff && tar -xJf {WORK}/ff.tar.xz -C {WORK}/ff --strip-components=1")
    ENV["PATH"] = f"{WORK}/ff:" + ENV["PATH"]
sh("node --version && npm --version && ffmpeg -version | head -1", log="toolchain.txt")
# Chrome system libraries. Installed one by one: on Ubuntu 24.04 several were renamed with a
# "t64" suffix (libasound2 -> libasound2t64), and one unknown name aborts a combined install.
sh("apt-get update -qq", log="toolchain.txt", check=False, timeout=900)
for pkg in ["libnss3", "libnspr4", "libatk1.0-0", "libatk-bridge2.0-0", "libcups2", "libdrm2", "libxkbcommon0",
            "libatspi2.0-0", "libxcomposite1", "libxdamage1", "libxfixes3", "libxrandr2", "libgbm1",
            "libpango-1.0-0", "libcairo2", "libasound2", "fonts-dejavu-core"]:
    rc, _ = sh(f"apt-get install -y -qq {pkg} > /dev/null", check=False, timeout=600)
    if rc:
        sh(f"apt-get install -y -qq {pkg}t64 > /dev/null", log="toolchain.txt", check=False, timeout=600)
sh("npx --yes hyperframes@latest browser ensure", log="toolchain.txt", check=False, timeout=900)
_, missing = sh("ldd $(find /root/.cache/hyperframes -name chrome-headless-shell -type f | head -1) | grep 'not found' || true",
                log="toolchain.txt", check=False)

os.makedirs(f"{assets}/fonts", exist_ok=True)
for name, url in (("Geist.woff2", "https://cdn.jsdelivr.net/npm/geist@1.3.1/dist/fonts/geist-sans/Geist-Variable.woff2"),
                  ("GeistMono.woff2", "https://cdn.jsdelivr.net/npm/geist@1.3.1/dist/fonts/geist-mono/GeistMono-Variable.woff2")):
    sh(f"curl -sfL -o {assets}/fonts/{name} {url}", log="toolchain.txt", check=False)
for f in glob.glob(f"{assets}/fonts/*"):                          # drop failed downloads
    if os.path.getsize(f) < 20_000:
        os.remove(f)
print("fonts", os.listdir(f"{assets}/fonts"), flush=True)
sh(f"curl -sL -o {assets}/gsap.min.js https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js")

# ---- composition
comp = f"{WORK}/composition"
shutil.rmtree(comp, ignore_errors=True)
sh(["python", builder, "--clips", clips, "--assets", assets, "--out", comp], log="build.txt")
with zipfile.ZipFile(f"{OUT}/composition.zip", "w", zipfile.ZIP_DEFLATED) as z:   # the page and timing, not the media
    for f in ("index.html", "timing.json"):
        z.write(f"{comp}/{f}", f)

# ---- hyperframes: lint, check, render
HF = "npx --yes hyperframes@latest"
sh(f"{HF} --version", log="toolchain.txt", check=False, cwd=comp)
lint_rc, _ = sh(f"{HF} lint", log="lint.txt", check=False, cwd=comp)
check_rc, _ = sh(f"{HF} check", log="check.txt", check=False, cwd=comp, timeout=2400)
render_rc, _ = sh(f"{HF} render --quality high --output {WORK}/brag.mp4", log="render.txt", check=False, cwd=comp,
                  timeout=4 * 3600)
if render_rc or not os.path.exists(f"{WORK}/brag.mp4"):
    json.dump({"lint_rc": lint_rc, "check_rc": check_rc, "render_rc": render_rc}, open(f"{OUT}/status.json", "w"))
    raise SystemExit("render failed; see render.txt / check.txt")

# ---- poster baked as frame 0, preview sheet
t = json.load(open(f"{comp}/timing.json"))
sh(f"ffmpeg -y -ss {t['poster_t']} -i {WORK}/brag.mp4 -frames:v 1 -q:v 2 {OUT}/poster.jpg", log="render.txt")
sh(f"ffmpeg -y -i {WORK}/brag.mp4 -i {OUT}/poster.jpg -filter_complex \"[0:v][1:v]overlay=0:0:enable='eq(n,0)'[v]\" "
   f"-map \"[v]\" -map 0:a? -c:v libx264 -crf 19 -preset slow -pix_fmt yuv420p -c:a aac -b:a 160k -movflags +faststart "
   f"{WORK}/silent.mp4", log="render.txt")
music = (glob.glob(f"{assets}/music/*.mp3") or glob.glob("/kaggle/input/**/music/*.mp3", recursive=True))[0]
D = t["duration"]
bed = (f"[1:a]aloop=loop=-1:size=2e9,atrim=0:{D},asetpts=N/SR/TB,volume=0.22,afade=t=in:d=1.5,"
       f"afade=t=out:st={D - 3.5}:d=3.5[m]")
has_audio = "audio" in sh(f"ffprobe -v error -show_entries stream=codec_type -of csv=p=0 {WORK}/silent.mp4", check=False)[1]
mix = (f"{bed};[0:a][m]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11[a]" if has_audio else
       f"{bed};[m]loudnorm=I=-16:TP=-1.5:LRA=11[a]")
sh(f'ffmpeg -y -i {WORK}/silent.mp4 -i "{music}" -filter_complex "{mix}" -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k '
   f"-ar 48000 -t {D} -movflags +faststart {OUT}/aisle_agent_demo.mp4", log="render.txt")
times = [round(2 + i * (t["duration"] - 3) / 19, 1) for i in range(20)]
for i, s in enumerate(times):
    sh(f"ffmpeg -y -ss {s} -i {OUT}/aisle_agent_demo.mp4 -frames:v 1 -vf scale=480:-1 {WORK}/p{i:02d}.jpg", check=False)
sh(f"ffmpeg -y -pattern_type glob -i '{WORK}/p*.jpg' -vf tile=4x5 -frames:v 1 -q:v 4 {OUT}/preview.jpg", check=False)
_, probe = sh(f"ffprobe -v error -show_entries format=duration:stream=codec_name,width,height -of json "
              f"{OUT}/aisle_agent_demo.mp4", check=False)
json.dump({"lint_rc": lint_rc, "check_rc": check_rc, "render_rc": render_rc, "probe": probe, "preview_times": times,
           "mb": round(os.path.getsize(f"{OUT}/aisle_agent_demo.mp4") / 1e6, 2)},
          open(f"{OUT}/status.json", "w"), indent=2)
print(open(f"{OUT}/status.json").read())
