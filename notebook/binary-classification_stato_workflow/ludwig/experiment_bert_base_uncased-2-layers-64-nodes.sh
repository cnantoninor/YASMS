#!/bin/bash

EXP_NAME="bert_base_uncased-2-layers-64-node"

ludwig experiment \
  --config ./${EXP_NAME}.yml \
  --training_set ./data/TESTNINO1_boolean_StatoWorkflow_train.csv \
  --test_set ./data/TESTNINO1_boolean_StatoWorkflow_test.csv \
  --experiment_name ${EXP_NAME}
