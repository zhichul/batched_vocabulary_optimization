#!/usr/bin/env bash


for HIDDEN in 384
do
for LAYER in 1
do
for HEAD in 1
do
for DROP in 0.05 0.1 0.2
do
for FIELD in train_loss avg_token
do
  python3 ../best_dev.py /export/a01/artifacts/bopt/simple/exp1-26/46/0.01/${HIDDEN}-${LAYER}-${HEAD}-${DROP}/log.json ${FIELD}
done
echo " "
done
done
done
done