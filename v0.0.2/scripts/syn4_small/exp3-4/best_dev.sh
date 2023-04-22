#!/usr/bin/env bash
EXPID="3-4"
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}

for SEED in 42
do
for SIZE in 768
do
for VSIZE in 50 100 200 400
do
for N in 10 15
do
for LR in 0.02 0.06 0.006
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${VSIZE}/${N}best/${LR}/None
python3 -m experiments.scripts.best_dev ${OUTPUT_DIR}/log.json dev_accuracy max
done
done
done
done
done
done
