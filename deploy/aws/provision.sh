#!/bin/bash
# Create the Aisle Agent host from nothing: key pair, security group, Elastic IP, a CPU instance
# (the vision-language model runs on Nebius Token Factory, so no GPU), first-boot setup via user-data.
# Run once from the repo root:  bash deploy/aws/provision.sh
# Then:                         ENGINE=nemotron HOST=<ip> bash deploy/aws/deploy.sh
set -euo pipefail
export MSYS_NO_PATHCONV=1                      # Git Bash on Windows: keep /dev/sda1 etc. as written
REGION=${REGION:-us-east-1}
TYPE=${TYPE:-c7i.xlarge}                       # 4 vCPU / 8 GB: pose tracking at ~27 fps (320 px) on CPU
NAME=${NAME:-aisle-agent}
PEM="$HOME/.ssh/$NAME-aws.pem"
AWS=(aws --region "$REGION")

if [ ! -f "$PEM" ]; then
  "${AWS[@]}" ec2 create-key-pair --key-name "$NAME" --query KeyMaterial --output text > "$PEM"
  chmod 600 "$PEM"
fi

SG=$("${AWS[@]}" ec2 describe-security-groups --filters "Name=group-name,Values=$NAME" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ -z "$SG" ] || [ "$SG" = "None" ]; then
  SG=$("${AWS[@]}" ec2 create-security-group --group-name "$NAME" --description "$NAME web + ssh" --query GroupId --output text)
  MYIP=$(curl -s https://checkip.amazonaws.com)
  "${AWS[@]}" ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null
  for p in 80 443; do
    "${AWS[@]}" ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port "$p" --cidr 0.0.0.0/0 >/dev/null
  done
  "${AWS[@]}" ec2 authorize-security-group-ingress --group-id "$SG" --protocol udp --port 10000-60000 --cidr 0.0.0.0/0 >/dev/null  # WebRTC media
fi

AMI=$("${AWS[@]}" ec2 describe-images --owners 099720109477       --filters "Name=name,Values=ubuntu/images/hvm-ssd*/ubuntu-jammy-22.04-amd64-server-*" "Name=state,Values=available"       --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)      # Canonical's latest Ubuntu 22.04
ALLOC=$("${AWS[@]}" ec2 allocate-address --domain vpc --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=$NAME}]" \
        --query AllocationId --output text)
IP=$("${AWS[@]}" ec2 describe-addresses --allocation-ids "$ALLOC" --query 'Addresses[0].PublicIp' --output text)
DOMAIN="${IP//./-}.sslip.io"

sed "s/__DOMAIN__/$DOMAIN/" deploy/aws/user-data.sh > deploy/aws/_ud.sh      # git-ignored
IID=$("${AWS[@]}" ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name "$NAME" \
      --security-group-ids "$SG" --user-data "file://deploy/aws/_ud.sh" \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3}' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" --query 'Instances[0].InstanceId' --output text)
"${AWS[@]}" ec2 wait instance-running --instance-ids "$IID"
"${AWS[@]}" ec2 associate-address --instance-id "$IID" --allocation-id "$ALLOC" >/dev/null
[ -f .aisle_key ] || python -c "import secrets; print(secrets.token_urlsafe(18))" > .aisle_key     # the shop key, git-ignored
printf 'REGION=%s\nINSTANCE=%s\nIP=%s\nDOMAIN=%s\nALLOC=%s\nSG=%s\n' "$REGION" "$IID" "$IP" "$DOMAIN" "$ALLOC" "$SG" > deploy/aws/.instance
echo "instance $IID at $IP -> https://$DOMAIN (first-boot setup takes ~8 min; then run deploy.sh)"
