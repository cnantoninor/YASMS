#!/bin/bash

# This script is an example of how to predict using curl.
# read the host name from the first argument and default to "localhost" if not passed
HOST=${1:-localhost}
TEXT=${2:-"FuoriSalone 2024"}

# if the name of the first parameter is prod or production use 85.235.146.174 as HOST
if [ "$HOST" == "prod" ] || [ "$HOST" == "production" ]; then
  HOST="85.235.146.174"
fi

# echo the calling parameters
echo "Calling predict endpoint with HOST=$HOST and TEXT=$TEXT"

curl -X 'POST' \
  "http://$HOST:8000/models/spam_classifier/GradientBoostingClassifier/test_nino/predict" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": [
    {
      "name": "text",
      "value": "$TEXT"
    }
  ]
}'

echo ""