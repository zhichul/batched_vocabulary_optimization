#!/usr/bin/env bash


for HIDDEN in 768
do
for FIELD in train_loss avg_token
do
for SEED in 42 44 46
do
  python3 ../best_dev.py /export/a01/artifacts/bopt/simple/exp1-24/${SEED}/0.01/${HIDDEN}/log.json ${FIELD}
done
done
echo " "
done
