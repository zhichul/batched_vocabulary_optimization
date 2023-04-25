#!/usr/bin/env bash
EXPID="23-1"
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}

for SIZE in 768
do
for VSIZE in 50 100 200 400
do
for N in 10 15
do
for SEED in 42 44 46
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${VSIZE}/${N}best/small
python3 -m experiments.scripts.best_dev ${OUTPUT_DIR}/log.json dev_accuracy max
done
done
done
done