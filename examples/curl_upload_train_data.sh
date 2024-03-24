#!/bin/bash

curl -X 'POST' \
  'http://85.235.146.174:8000/models/spam_classifier/GradientBoostingClassifier/test_nino/upload_train_data' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'train_data=@model_data.csv;type=text/csv' \
  -F 'features_fields=Testo' \
  -F 'target_field=Stato Workflow'