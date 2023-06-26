#!/usr/bin/env bash
EXPID="22"
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
for SEED in 42
do
for SIZE in 768
do
for L1 in 0.01 0.1
do
for DATA in 500
do
for MIXTURE in 0.9875 0.975 0.95 0.9 0.8 0.6
do
for INPUT_NAME in dev test
do

DATA_PREFIX=${BLU_CORPORA}/vopt/syn/4/${DATA}
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/${MIXTURE}
LOAD_DIR=${BLU_ARTIFACTS}/boptv2/syn4_small/exp21-1/${SEED}/${SIZE}/${L1}/${DATA}/checkpoint-final
python
done
done
done
done
done
done