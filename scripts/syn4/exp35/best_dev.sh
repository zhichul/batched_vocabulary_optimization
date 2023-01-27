#!/usr/bin/env bash

for SEED in 42
do
for LAYER in 1 2
do
for HEAD in 1 2 4
do
for SIZE in 24 96 192
do
for VSIZE in 50 100 200 400
do
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn4/exp35/42/${LAYER}/${HEAD}/${SIZE}/${VSIZE}/log.json eval_zero_one_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn4/exp35/42/${LAYER}/${HEAD}/${SIZE}/${VSIZE}/log.json eval_expected_zero_one_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn4/exp35/42/${LAYER}/${HEAD}/${SIZE}/${VSIZE}/log.json eval_log_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn4/exp35/42/${LAYER}/${HEAD}/${SIZE}/${VSIZE}/log.json train_loss
echo "##########################"
done
done
done
done
done