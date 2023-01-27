#!/usr/bin/env bash

for SEED in 42
do
for LAYER in 1 2
do
for HEAD in 1 2 4
do
for SIZE in 24 96 192
do
for VSIZE in 200 400
do
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn3/exp5/42/${LAYER}/${HEAD}/${SIZE}/${VSIZE}/log.json zero_one_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn3/exp5/42/${LAYER}/${HEAD}/${SIZE}/${VSIZE}/log.json expected_zero_one_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn3/exp5/42/${LAYER}/${HEAD}/${SIZE}/${VSIZE}/log.json log_loss
python3 ../../simple/analysis/best_dev.py ${BLU_ARTIFACTS}/bopt/syn3/exp5/42/${LAYER}/${HEAD}/${SIZE}/${VSIZE}/log.json train_loss
echo "##########################"
done
done
done
done
done