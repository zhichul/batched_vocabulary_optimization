#!/usr/bin/env bash
EXPID="1-2"
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}

for SEED in 42 44 46
do
for L1 in 0.1 1.0 0.01
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${L1}
python3 -m experiments.scripts.best_dev ${OUTPUT_DIR}/log.json dev_accuracy max
done
done
