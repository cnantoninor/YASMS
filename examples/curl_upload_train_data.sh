#!/bin/bash

# This script is an example of how to upload train data to a model using curl.
# read the host name from the first argument and default to "localhost" if not passed
HOST=${1:-localhost}

# if the name of the first parameter is prod or production use 85.235.146.174 as HOST
if [ "$HOST" == "prod" ] || [ "$HOST" == "production" ]; then
  HOST="85.235.146.174"
fi

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
TEST_TRAIN_DATA=$SCRIPT_DIR/../test_data/MDLC/spam_classifier/test_model/test_project/DATA_UPLOADED/model_data.csv

curl -X 'POST' \
  "http://$HOST:8000/models/spam_classifier/GradientBoostingClassifier/test_nino/upload_train_data" \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'features_fields=Testo' \
  -F 'target_field=Stato Workflow' \
  -F "train_data=@$TEST_TRAIN_DATA" \
  -F 'features_fields=Testo' \
  -F 'target_field=Stato Workflow'