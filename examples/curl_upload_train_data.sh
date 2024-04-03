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
# TEST_TRAIN_DATA=$SCRIPT_DIR/../test_data/MDLC/spam_classifier/test_model/test_project/DATA_UPLOADED/model_data.csv
# TEST_TRAIN_DATA="/home/arau6/projects/wl-semsearch-poc-debug/test_data/23457ec5-79c6-4542-a14a-14a3c96d90cb.csv"
TEST_TRAIN_DATA="/home/arau6/projects/wl-semsearch-poc-debug/remote/85.235.146.174/data/spam_classifier/GradientBoostingClassifier/lodes/20240329_17-37-12-677202/lodes_model_data.csv"
# check if TEST_TRAIN_DATA exists
if [ ! -f "$TEST_TRAIN_DATA" ]; then
  echo "The file $TEST_TRAIN_DATA does not exist."
  exit 1
fi

# curl -X 'POST' \
#   "http://$HOST:8000/models/spam_classifier/GradientBoostingClassifier/test_nino/upload_train_data" \
#   -H 'accept: application/json' \
#   -H 'Content-Type: multipart/form-data' \
#   -F 'features_fields=Testo' \
#   -F 'target_field=Stato Workflow' \
#   -F "train_data=@$TEST_TRAIN_DATA" \

curl -X 'POST' \
  "http://$HOST:8000/models/spam_classifier/GradientBoostingClassifier/test_nino/upload_train_data" \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'features_fields=text' \
  -F 'target_field=status' \
  -F "train_data=@$TEST_TRAIN_DATA" \
