#!/usr/bin/env bash
for VSIZE in 50 100 134 150 200 400 600 800 853
do
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn4/exp2/42/768/${VSIZE}/log.json zero_one_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn4/exp2/42/768/${VSIZE}/log.json expected_zero_one_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn4/exp2/42/768/${VSIZE}/log.json log_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn4/exp2/42/768/${VSIZE}/log.json train_loss
done