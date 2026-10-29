#!/bin/bash
# Create (or update) the wake link: a public Lambda function URL that starts the stopped instance and
# forwards to the app once it answers. Run from the repo root after provision.sh:
#   bash deploy/aws/wake.sh
# The role can only describe instances and start this one. Writes WAKE_URL to deploy/aws/.instance.
set -euo pipefail
source deploy/aws/.instance
NAME=${WAKE_NAME:-aisle-agent-wake}
export MSYS_NO_PATHCONV=1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

if ! aws iam get-role --role-name "$NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$NAME" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$NAME" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  sleep 10                                                       # let the new role propagate
fi
aws iam put-role-policy --role-name "$NAME" --policy-name start-demo-instance --policy-document "{
  \"Version\":\"2012-10-17\",\"Statement\":[
    {\"Effect\":\"Allow\",\"Action\":\"ec2:DescribeInstances\",\"Resource\":\"*\"},
    {\"Effect\":\"Allow\",\"Action\":\"ec2:StartInstances\",\"Resource\":\"arn:aws:ec2:$REGION:$ACCOUNT:instance/$INSTANCE\"}]}"

tmp=$(mktemp -d)
python -m zipfile -c "$tmp/wake.zip" deploy/aws/wake/lambda_function.py
ENV="Variables={INSTANCE=$INSTANCE,APP_URL=https://$DOMAIN}"
if aws lambda get-function --region "$REGION" --function-name "$NAME" >/dev/null 2>&1; then
  aws lambda update-function-code --region "$REGION" --function-name "$NAME" --zip-file "fileb://$tmp/wake.zip" >/dev/null
  aws lambda wait function-updated --region "$REGION" --function-name "$NAME"
  aws lambda update-function-configuration --region "$REGION" --function-name "$NAME" --environment "$ENV" >/dev/null
else
  for i in 1 2 3 4 5; do                                         # a brand-new role can take a few seconds to be assumable
    aws lambda create-function --region "$REGION" --function-name "$NAME" --runtime python3.12 \
      --handler lambda_function.lambda_handler --role "arn:aws:iam::$ACCOUNT:role/$NAME" \
      --zip-file "fileb://$tmp/wake.zip" --timeout 15 --memory-size 128 --environment "$ENV" >/dev/null && break
    sleep 8
  done
  aws lambda wait function-active-v2 --region "$REGION" --function-name "$NAME"
  aws lambda create-function-url-config --region "$REGION" --function-name "$NAME" --auth-type NONE >/dev/null
  aws lambda add-permission --region "$REGION" --function-name "$NAME" --statement-id public-url \
    --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE >/dev/null
  aws lambda add-permission --region "$REGION" --function-name "$NAME" --statement-id public-url-invoke \
    --action lambda:InvokeFunction --principal "*" --invoked-via-function-url >/dev/null 2>&1 || true
fi
rm -rf "$tmp"
URL=$(aws lambda get-function-url-config --region "$REGION" --function-name "$NAME" --query FunctionUrl --output text)
grep -v '^WAKE_URL=' deploy/aws/.instance > deploy/aws/.instance.new && echo "WAKE_URL=$URL" >> deploy/aws/.instance.new
mv deploy/aws/.instance.new deploy/aws/.instance
echo "wake link: $URL"
