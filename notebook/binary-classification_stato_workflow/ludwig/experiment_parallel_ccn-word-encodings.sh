#!/bin/bash
# FILEPATH: /home/arau6/projects/wl-semsearch-poc/notebook/binary-classification_stato_workflow/ludwig/experiment_parallel_ccn-word-encodings.sh

ludwig experiment \
  --dataset ../../../src/data/wl_classif_testnino1/TESTNINO1_with_boolean_StatoWorkflow.csv  \
  --config ./ludwig-parallel-cnn-word-encoding.yml \
  --experiment_name parallel-cnn-word-encoding-balanced-train
