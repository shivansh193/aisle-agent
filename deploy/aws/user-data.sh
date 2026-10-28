#!/bin/bash
# EC2 user-data for the Aisle Agent live server (Ubuntu 22.04; CPU PyTorch by default, the vision-language model is hosted).
# Installs Caddy (automatic HTTPS for DOMAIN) and the Python environment. The app itself is copied
# and started by deploy.sh. Progress: /var/log/cloud-init-output.log; done when /opt/ss/READY exists.
set -euxo pipefail
DOMAIN="__DOMAIN__"

apt-get update -y
apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl gnupg ffmpeg libgl1 python3-venv
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
apt-get update -y && apt-get install -y caddy
cat > /etc/caddy/Caddyfile <<EOF
${DOMAIN} {
  encode gzip
  reverse_proxy 127.0.0.1:8000 {
    flush_interval -1
  }
}
EOF
systemctl restart caddy

mkdir -p /opt/ss/app /opt/ss/cache
python3 -m venv /opt/ss/venv
/opt/ss/venv/bin/pip install -q --upgrade pip
/opt/ss/venv/bin/pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/${TORCH_INDEX:-cpu}
/opt/ss/venv/bin/pip install -q transformers==4.51.3 accelerate==1.6.0 ultralytics==8.3.130 lap==0.5.12 aiortc==1.10.1 \
  fastapi==0.115.12 uvicorn==0.34.2 websockets==13.1 httpx==0.28.1 python-multipart==0.0.20 opencv-python-headless==4.10.0.84 pillow "numpy<2.3"
cd /opt/ss && /opt/ss/venv/bin/python -c "from ultralytics import YOLO; YOLO('/opt/ss/yolo11n-pose.pt')" || true
chown -R ubuntu:ubuntu /opt/ss
touch /opt/ss/READY
