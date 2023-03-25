#!/usr/bin/env bash

python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/sentiment140/expr1-1/42/0.1/768/16000/log1.json eval_zero_one_loss test_zero_one_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/sentiment140/expr1-1/42/0.1/768/16000/log1.json eval_expected_zero_one_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/sentiment140/expr1-1/42/0.1/768/16000/log1.json eval_log_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/sentiment140/expr1-1/42/0.1/768/16000/log1.json train_loss