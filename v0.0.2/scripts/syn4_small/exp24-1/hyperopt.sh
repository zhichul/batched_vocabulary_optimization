#!/usr/bin/env bash
source vars.sh
source ../bash_scripts/hyperopt.sh
METRIC=dev_accuracy
ORDER=max
CHECKPOINT=checkpoint-early-stopping
RESULTS_FILE=hyperopt-results-early-stopping-lattice.json
lattice_hyperopt
json_extract step test_accuracy test_boundary_precision test_boundary_recall test_boundary_f1  < ${RESULTS_FILE}
json_extract test_accuracy test_boundary_precision test_boundary_recall test_boundary_f1 --sep " & " --tail "\\\\"  --format "{:.2f}" "{:.2f}" "{:.2f}" "{:.2f}" --preprocess x100 x100 x100 x100 < ${RESULTS_FILE}

METRIC=step
ORDER=max
CHECKPOINT=checkpoint-final
RESULTS_FILE=hyperopt-results-final-iteration-lattice.json
lattice_hyperopt
json_extract step test_accuracy test_boundary_precision test_boundary_recall test_boundary_f1  < ${RESULTS_FILE}
json_extract test_accuracy test_boundary_precision test_boundary_recall test_boundary_f1 --sep " & " --tail "\\\\" --format "{:.2f}" "{:.2f}" "{:.2f}" "{:.2f}" --preprocess x100 x100 x100 x100 < ${RESULTS_FILE}
