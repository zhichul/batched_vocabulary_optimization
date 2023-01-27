#!/usr/bin/env bash

EXPID=22
DATA_PREFIX=${BLU_CORPORA}/vopt/syn/3/full
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/syn3/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/syn3/exp${EXPID}
for SEED in 42
do
for LAYER in 1 2
do
for HEAD in 1 2 4
do
for SIZE in 24 96 192
do
for L1 in gold
do
python3 ../../simple/analysis/best_dev.py ${ARTIFACT_PREFIX}/${SEED}/${LAYER}/${HEAD}/${SIZE}/${L1}/log.json zero_one_loss
python3 ../../simple/analysis/best_dev.py ${ARTIFACT_PREFIX}/${SEED}/${LAYER}/${HEAD}/${SIZE}/${L1}/log.json expected_zero_one_loss
python3 ../../simple/analysis/best_dev.py ${ARTIFACT_PREFIX}/${SEED}/${LAYER}/${HEAD}/${SIZE}/${L1}/log.json log_loss
python3 ../../simple/analysis/best_dev.py ${ARTIFACT_PREFIX}/${SEED}/${LAYER}/${HEAD}/${SIZE}/${L1}/log.json train_loss
done
done
done
done
done