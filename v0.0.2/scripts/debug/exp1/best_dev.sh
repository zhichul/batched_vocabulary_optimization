#!/usr/bin/env bash
for L1 in 0.01 0.1 1.0
do
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn3/exp34-11/42/${L1}/768/log.json eval_zero_one_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn3/exp34-11/42/${L1}/768/log.json eval_expected_zero_one_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn3/exp34-11/42/${L1}/768/log.json eval_log_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn3/exp34-11/42/${L1}/768/log.json train_loss
done