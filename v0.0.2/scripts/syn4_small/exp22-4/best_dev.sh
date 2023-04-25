#!/usr/bin/env bash
EXPID="22-4"
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}

for SIZE in 768
do
for DATA in 100
do
for VSIZE in 50
do
for SEED in 42 44 46
do
for DROPOUT in 0.0 0.1 0.2 0.4
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${VSIZE}/${DATA}/${DROPOUT}
python3 -m experiments.scripts.best_dev ${OUTPUT_DIR}/log.json dev_accuracy max
done
done
done
echo ""
done
done
