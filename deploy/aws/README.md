# Aisle Agent on AWS (EC2, one T4 GPU)

What runs: a `g4dn.xlarge` (NVIDIA T4) from the Deep Learning Base OSS Nvidia Driver AMI
(Ubuntu 22.04), Caddy for automatic HTTPS on `<ip-with-dashes>.sslip.io`, and the server as the
`aisle-agent` systemd service. About $0.53/hour while running in us-east-1; a stopped instance costs
only its 100 GB disk and the Elastic IP (a few dollars a month).

- `user-data.sh`: first-boot setup (Caddy, Python venv with the CUDA build of PyTorch, YOLO weights).
- `deploy.sh`: copies `aisle_agent` and the vjev package, writes `/etc/aisle-agent.env`
  (shop key from `.aisle_key`, Ring credentials from `.ring_env`), restarts the service.

```bash
HOST=184.196.182.169 bash deploy/aws/deploy.sh                     # redeploy after a change
aws ec2 stop-instances  --region us-east-1 --instance-ids i-0438ab7e5490c3842   # pause billing
aws ec2 start-instances --region us-east-1 --instance-ids i-0438ab7e5490c3842   # resume (same IP)
ssh -i ~/.ssh/aisle-agent-aws.pem ubuntu@184.196.182.169 journalctl -u aisle-agent -f   # logs
```

Ring app URLs for this host: `https://184-196-182-169.sslip.io/ring/oauth/token` (Token Exchange),
`/ring/link` (Account Link), `/v1/ring/webhook` (Webhook).
