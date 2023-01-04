#!/usr/bin/env bash


for HIDDEN in 768 384 96
do
for LAYER in 8 4 1
do
for HEAD in 12 6 1
do
for FIELD in train_loss avg_token
do
  python3 ../best_dev.py /export/a01/artifacts/bopt/simple/exp2-16-d/42/${HIDDEN}-${LAYER}-${HEAD}/log.json ${FIELD}
done
echo " "
done
done
done