#!/usr/bin/env bash
EXPID="3-2"
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}

for SEED in 42
do
for SIZE in 768
do
for VSIZE in 50 100 200 400
do
for N in 3 5 10 15
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${VSIZE}/${N}best/
python3 -m experiments.scripts.best_dev ${OUTPUT_DIR}/log.json dev_accuracy max
done
done
done
done