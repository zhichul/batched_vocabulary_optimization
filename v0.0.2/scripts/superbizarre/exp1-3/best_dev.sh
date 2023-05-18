#!/usr/bin/env bash
EXPID="1-3"
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}

for SEED in 42 44 46
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/
python3 -m experiments.scripts.best_dev ${OUTPUT_DIR}/log.json dev_accuracy max
done
