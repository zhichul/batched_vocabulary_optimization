#!/usr/bin/env bash
EXPID="22"
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}

for SIZE in 768
do
for DATA in 100 500 small full
do
for VSIZE in 50 100 200 400
do
for SEED in 42 44 46
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${VSIZE}/${DATA}
python3 -m experiments.scripts.best_dev ${OUTPUT_DIR}/log.json dev_accuracy max
done
done
echo ""
done
done
