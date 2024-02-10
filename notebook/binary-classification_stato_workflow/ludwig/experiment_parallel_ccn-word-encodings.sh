#!/bin/bash

ludwig experiment \
  --config ./parallel-cnn-word-encoding.yml \
  --training_set ./data/TESTNINO1_boolean_StatoWorkflow_train.csv \
  --test_set ./data/TESTNINO1_boolean_StatoWorkflow_test.csv \
  --experiment_name parallel-cnn-word-encoding-balanced-train
