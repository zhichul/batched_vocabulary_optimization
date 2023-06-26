#!/usr/bin/env bash
source vars.sh
source ../bash_scripts/hyperopt.sh
METRIC=dev_accuracy
ORDER=max
CHECKPOINT=checkpoint-early-stopping
RESULTS_FILE=hyperopt-results-early-stopping-lattice.json
lattice_full_attention_hyperopt
json_extract step test_accuracy  < ${RESULTS_FILE}
json_extract test_accuracy  --sep " & " --tail "\\\\"  --format "{:.2f}" --preprocess x100 < ${RESULTS_FILE}

METRIC=step
ORDER=max
CHECKPOINT=checkpoint-final
RESULTS_FILE=hyperopt-results-final-iteration-lattice.json
lattice_full_attention_hyperopt
json_extract step test_accuracy  < ${RESULTS_FILE}
json_extract test_accuracy  --sep " & " --tail "\\\\"  --format "{:.2f}" --preprocess x100 < ${RESULTS_FILE}