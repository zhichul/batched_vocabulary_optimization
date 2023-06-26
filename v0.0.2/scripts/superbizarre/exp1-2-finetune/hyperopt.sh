#!/usr/bin/env bash
source vars.sh
source ../bash_scripts/hyperopt.sh
bert_finetune_hyperopt
json_extract step test_accuracy  < hyperopt-results.json
json_extract test_accuracy --sep " & " --tail "\\\\"  --format "{:.2f}" --preprocess  x100 < hyperopt-results.json
