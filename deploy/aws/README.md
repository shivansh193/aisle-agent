# Aisle Agent on AWS: a CPU server and Nebius Token Factory

The vision-language model and the agents run on NVIDIA Nemotron through Nebius Token Factory, so
the server needs no GPU. Pose tracking is the only model that runs on every frame, at about 17 fps
on 4 vCPU at 416 px.

What runs:
- a `c7i.xlarge` (4 vCPU, 8 GB, Ubuntu 22.04) in us-east-1, about $0.18/hour while running;
- Caddy for automatic HTTPS on `<ip-with-dashes>.sslip.io`;
- the server as the `aisle-agent` systemd service with `--engine nemotron`.

Stopped, it costs only its 30 GB disk and the Elastic IP. With the wake link and idle stop, it runs only while someone is
using it: share the wake link, not the app URL.

| Script | What it does |
|---|---|
| `provision.sh` | Creates the key pair (`~/.ssh/aisle-agent-aws.pem`), the security group (SSH from your IP; 80/443 and WebRTC UDP open), an Elastic IP and the instance, with `user-data.sh` as first-boot setup. Writes the ids to `deploy/aws/.instance` and a shop key to `.aisle_key` (both git-ignored). |
| `user-data.sh` | First boot: Caddy, Python venv with CPU PyTorch, YOLO weights. Done when `/opt/ss/READY` exists (~8 min). |
| `deploy.sh` | Copies `aisle_agent` and `vjev`, writes `/etc/aisle-agent.env` (root-only: shop key, `TOKEN_FACTORY_API_KEY` from `.env`), restarts the service. Also turns on the Caddy access log and the idle stop (`idle_stop.py`, checked every minute): the instance shuts itself down after `IDLE_MIN` (15) minutes without a request, or `PAGE_MIN` (60) minutes without a page being opened. `IDLE_MIN=0` turns it off. |
| `wake.sh` | Creates the wake link: a public Lambda function URL (`wake/lambda_function.py`) that starts the stopped instance, shows a waiting page and forwards to the app once `/health` answers (about a minute). Its role can only describe instances and start this one. Writes `WAKE_URL` to `.instance`. |

```bash
bash deploy/aws/provision.sh                                   # once
source deploy/aws/.instance
ENGINE=nemotron HOST=$IP bash deploy/aws/deploy.sh             # deploy / redeploy
ssh -i ~/.ssh/aisle-agent-aws.pem ubuntu@$IP journalctl -u aisle-agent -f      # logs
aws ec2 stop-instances  --region $REGION --instance-ids $INSTANCE              # pause billing
aws ec2 start-instances --region $REGION --instance-ids $INSTANCE              # resume (same IP)
```

For a GPU host (local Qwen engine, or SigLIP photo matching), use a `g4dn.xlarge` with a CUDA
image: `TYPE=g4dn.xlarge bash deploy/aws/provision.sh`, and edit `user-data.sh` to use the CUDA
PyTorch index (`cu121` in place of `cpu`).
