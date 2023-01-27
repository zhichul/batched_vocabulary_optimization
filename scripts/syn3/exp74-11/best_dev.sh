#!/usr/bin/env bash
for L1 in 1.0  0.1  0.01
do
for LR in 0.2 0.06 0.02 0.006 0.002 0.0006 0.0002 0.00006
do
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn3/exp74-11/42/768/${L1}/${LR}/log.json test_tok_f1 test_zero_one_loss test_leakage test_over_attention_mean test_over_attention_count test_over_attention_mass test_total_attention_dist_count
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn3/exp74-11/42/768/${L1}/${LR}/log.json test_tok_marginal
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn3/exp74-11/42/768/${L1}/${LR}/log.json test_path_marginal
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn3/exp74-11/42/768/${L1}/${LR}/log.json test_path_marginal
done
done
