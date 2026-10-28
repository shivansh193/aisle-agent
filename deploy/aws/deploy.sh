#!/bin/bash
# Copy the app to the EC2 instance and (re)start it as a systemd service. Run from the repo root:
#   HOST=<instance ip> bash deploy/aws/deploy.sh
#   ENGINE=nemotron ...   NVIDIA Nemotron VL via Nebius Token Factory (needs TOKEN_FACTORY_API_KEY in .env)
# Needs ~/.ssh/aisle-agent-aws.pem, .aisle_key (the shop key) and .env (TOKEN_FACTORY_API_KEY).
set -euo pipefail
HOST=${HOST:?set HOST to the instance IP}
KEY=${SSH_KEY:-$HOME/.ssh/aisle-agent-aws.pem}
VJEV_SRC=${VJEV_SRC:-vjev}
ENGINE=${ENGINE:-nemotron}
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@$HOST")

tmp=$(mktemp -d)
cp -r aisle_agent "$tmp/" && cp -r "$VJEV_SRC" "$tmp/vjev"
find "$tmp" -name __pycache__ -prune -exec rm -rf {} +
tar -C "$tmp" -czf "$tmp/app.tgz" aisle_agent vjev
scp -i "$KEY" -o StrictHostKeyChecking=accept-new "$tmp/app.tgz" "ubuntu@$HOST:/tmp/app.tgz"

{
  echo "AISLE_KEY=$(cat .aisle_key)"
  [ -f .env ] && grep -E '^(TOKEN_FACTORY_API_KEY|NEBIUS_API_KEY)=' .env || true
} | "${SSH[@]}" "sudo install -m 600 -o root -g root /dev/stdin /etc/aisle-agent.env"

"${SSH[@]}" "export ENGINE=$ENGINE; "'set -e
test -f /opt/ss/READY || { echo "instance still setting up (see /var/log/cloud-init-output.log)"; exit 3; }
rm -rf /opt/ss/app.new && mkdir -p /opt/ss/app.new && tar -C /opt/ss/app.new -xzf /tmp/app.tgz
rm -rf /opt/ss/app && mv /opt/ss/app.new /opt/ss/app
sudo tee /etc/systemd/system/aisle-agent.service >/dev/null <<EOF
[Unit]
Description=Aisle Agent live server
After=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/opt/ss
EnvironmentFile=/etc/aisle-agent.env
Environment=PYTHONPATH=/opt/ss/app HF_HOME=/opt/ss/cache/hf AISLE_LAYOUT=/opt/ss/shop_layout.json
ExecStart=/opt/ss/venv/bin/python -m aisle_agent.server --engine $ENGINE --preset fast --host 127.0.0.1 --port 8000 --pose-weights /opt/ss/yolo11n-pose.pt
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now aisle-agent && sudo systemctl restart aisle-agent
echo deployed'
rm -rf "$tmp"
