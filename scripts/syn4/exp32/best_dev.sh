#!/usr/bin/env bash
for VSIZE in 50 100 200 400
do
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn4/exp32/42/768/${VSIZE}/log.json eval_zero_one_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn4/exp32/42/768/${VSIZE}/log.json eval_expected_zero_one_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn4/exp32/42/768/${VSIZE}/log.json eval_log_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn4/exp32/42/768/${VSIZE}/log.json train_loss
done