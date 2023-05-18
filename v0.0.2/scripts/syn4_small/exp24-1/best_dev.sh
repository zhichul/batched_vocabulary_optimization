#!/usr/bin/env bash
EXPID="24-1"
mkdir -p ${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}

for SIZE in 768
do
for DATA in 100 500 small full
do
for L1 in 0.01 0.1 1.0
do
for SEED in 42 44 46 48 50 52 54 56 58 60
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}
python3 -m experiments.scripts.best_dev ${OUTPUT_DIR}/log.json dev_accuracy max
done
done
echo ""
done
done
