#!/usr/bin/env bash
source vars.sh
source ../bash_scripts/hyperopt.sh
bert_hyperopt
json_extract step test_accuracy test_boundary_precision test_boundary_recall test_boundary_f1  < hyperopt-results.json
json_extract test_accuracy test_boundary_precision test_boundary_recall test_boundary_f1 --sep " & " --tail "\\\\"  --format "{:.2f}" "{:.2f}" "{:.2f}" "{:.2f}" --preprocess x100 x100 x100 x100 < hyperopt-results.json
