#!/usr/bin/env bash
for size in 768 "2-4-192"
do
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/snli_1.0/expr1-3/42/0.1/${size}/16000/log1.json eval_zero_one_loss test_zero_one_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/snli_1.0/expr1-3/42/0.1/${size}/16000/log1.json eval_expected_zero_one_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/snli_1.0/expr1-3/42/0.1/${size}/16000/log1.json eval_log_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/snli_1.0/expr1-3/42/0.1/${size}/16000/log1.json train_loss
done