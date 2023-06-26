#!/usr/bin/env bash
source vars.sh

# this function prints the best iteration based on dev accuracy of log.json for each run
ARTIFACT_PREFIX=${BLU_ARTIFACTS2}/boptv2/syn4_small/exp${EXPID}

for SIZE in 768
do
for DATA in 100
do
for L1 in 0.01
do
for LOAD_DATA in 100 500 small full
do
for SEED in 42
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/learned-${LOAD_DATA}
python3 -m experiments.scripts.best_dev ${OUTPUT_DIR}/log.json dev_accuracy max test_accuracy
done
done
done
done
done
