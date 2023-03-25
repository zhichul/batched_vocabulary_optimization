#!/usr/bin/env bash
EXPID="20-sweep"
mkdir -p ${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/ptb
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/ptb/exp${EXPID}
for SEED in 44
do
for SIZE in 768
do
for LR in 0.02
do
for L1 in 0.1
do
for DROP in 0.1 0.2 0.4 0.8
do

python3 ../../simple/analysis/best_dev.py ${ARTIFACT_PREFIX}/${SEED}/${DROP}/${LR}/${L1}/log.json eval_avg_token train_loss test_avg_token

done
done
done
done
done
