#!/usr/bin/env bash

EXP=1-16
DIR=2022-11-8

python3 ../training_curves.py \
  $BLU_ARTIFACTS/bopt/simple/exp${EXP}/42/10.0/768/log.json \
  $BLU_ARTIFACTS/bopt/simple/exp${EXP}/42/100.0/768/log.json \
  --out losses.png \
  --field train_loss avg_token \
  --field_pretty train dev \
  --colors orange red \
  --linestyles solid dashed \
  --names [vopt-fp-fa-ent10] [vopt-fp-fa-ent100] \
  --xlabel "steps x100" \
  --ylabel "loss" \

for ENT in 10.0 100.0
do
  # extract the entropy
  cat /Users/blu/artifacts/bopt/simple/exp${EXP}/42/${ENT}/768/full_lattice_log.train.json | python3 ../extract_log.py ent > /Users/blu/artifacts/bopt/simple/exp${EXP}/42/${ENT}/768/ent.train.json
  cat /Users/blu/artifacts/bopt/simple/exp${EXP}/42/${ENT}/768/full_lattice_log.valid.json | python3 ../extract_log.py ent > /Users/blu/artifacts/bopt/simple/exp${EXP}/42/${ENT}/768/ent.valid.json

done

python3 ../training_curves.py \
  $BLU_ARTIFACTS/bopt/simple/exp${EXP}/42/10.0/768/ent.train.json \
  $BLU_ARTIFACTS/bopt/simple/exp${EXP}/42/10.0/768/ent.valid.json \
  $BLU_ARTIFACTS/bopt/simple/exp${EXP}/42/100.0/768/ent.train.json \
  $BLU_ARTIFACTS/bopt/simple/exp${EXP}/42/100.0/768/ent.valid.json \
  --out entropy.png \
  --field ent \
  --field_pretty ent \
  --colors orange cyan red blue \
  --linestyles solid \
  --names 10-train 10-valid 100-train 100-valid \
  --xlabel "steps x100" \
  --ylabel "entropy (per sentence)"

for ENT in 10.0 100.0
do
  # extract the units marginals and log_probs
  for DAT in train valid
  do
    cat /Users/blu/artifacts/bopt/simple/exp${EXP}/42/${ENT}/768/full_lattice_log.${DAT}.json | python3 ../extract_log.py marginal unit log_prob lm_marginal > /Users/blu/artifacts/bopt/simple/exp${EXP}/42/${ENT}/768/lattice.${DAT}.json
    python3 ../serialize_lattice.py /Users/blu/artifacts/bopt/simple/exp${EXP}/42/${ENT}/768/lattice.${DAT}.json lattices/${ENT}/${DAT}/
  done
done

rm -rf lattices/10
rm -rf lattices/100
mv lattices/10.0 lattices/10
mv lattices/100.0 lattices/100