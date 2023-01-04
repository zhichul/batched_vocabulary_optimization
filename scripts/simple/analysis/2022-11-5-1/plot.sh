#!/usr/bin/env bash

EXP=1-14
DIR=2022-11-5-1

python3 ../training_curves.py \
  $BLU_ARTIFACTS/bopt/simple/exp${EXP}/42/0.0/768/log.json \
  $BLU_ARTIFACTS/bopt/simple/exp2-12/44/768/log.json \
  $BLU_ARTIFACTS/bopt/simple/exp2-12-d/44/768/log.json \
  --out losses.png \
  --field train_loss avg_token \
  --field_pretty train dev \
  --colors orange cyan blue \
  --linestyles solid dashed \
  --names [vopt-fp] [uni] [uni-d] \
  --xlabel "steps x50" \
  --ylabel "loss" \
# extract the entropy
cat /Users/blu/artifacts/bopt/simple/exp${EXP}/42/0.0/768/full_lattice_log.train.json | python3 ../extract_log.py ent > /Users/blu/artifacts/bopt/simple/exp${EXP}/42/0.0/768/ent.train.json 
cat /Users/blu/artifacts/bopt/simple/exp${EXP}/42/0.0/768/full_lattice_log.valid.json | python3 ../extract_log.py ent > /Users/blu/artifacts/bopt/simple/exp${EXP}/42/0.0/768/ent.valid.json

python3 ../training_curves.py \
  $BLU_ARTIFACTS/bopt/simple/exp${EXP}/42/0.0/768/ent.train.json \
  $BLU_ARTIFACTS/bopt/simple/exp${EXP}/42/0.0/768/ent.valid.json \
  --out entropy.png \
  --field ent \
  --field_pretty ent \
  --colors orange cyan \
  --linestyles solid \
  --names train valid \
  --xlabel "steps x50" \
  --ylabel "entropy (per sentence)"

# extract the units marginals and log_probs
for DAT in train valid
do
  cat /Users/blu/artifacts/bopt/simple/exp${EXP}/42/0.0/768/full_lattice_log.${DAT}.json | python3 ../extract_log.py marginal unit log_prob lm_marginal > /Users/blu/artifacts/bopt/simple/exp${EXP}/42/0.0/768/lattice.${DAT}.json
  python3 ../serialize_lattice.py /Users/blu/artifacts/bopt/simple/exp${EXP}/42/0.0/768/lattice.${DAT}.json lattices/${DAT}/
done