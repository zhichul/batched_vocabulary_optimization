#!/usr/bin/env bash
for L1 in 1.0  0.1  0.01
do
for LR in 0.02 0.006 0.002 0.0006 0.0002 0.00006
do
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn3/exp74-12/42/768/${L1}/${LR}/log.json eval_tok_f1
done
done