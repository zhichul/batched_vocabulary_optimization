#!/usr/bin/env bash

EXP=1-13
DIR=2022-11-4-buggy-init
# extract the entropy
cat /Users/blu/artifacts/bopt/simple/exp${EXP}/42/0.0/768-buggy-init/full_lattice_log.train.json | python3 ../extract_log.py ent > /Users/blu/artifacts/bopt/simple/exp${EXP}/42/0.0/768-buggy-init/ent.train.json 
cat /Users/blu/artifacts/bopt/simple/exp${EXP}/42/0.0/768-buggy-init/full_lattice_log.valid.json | python3 ../extract_log.py ent > /Users/blu/artifacts/bopt/simple/exp${EXP}/42/0.0/768-buggy-init/ent.valid.json

python3 ../training_curves.py \
  $BLU_ARTIFACTS/bopt/simple/exp${EXP}/42/0.0/768-buggy-init/ent.train.json \
  $BLU_ARTIFACTS/bopt/simple/exp${EXP}/42/0.0/768-buggy-init/ent.valid.json \
  --out entropy.png \
  --field ent \
  --field_pretty ent \
  --colors orange cyan \
  --linestyles solid \
  --names train valid \
  --xlabel "steps x10" \
  --ylabel "entropy (per sentence)"

# extract the units marginals and log_probs
for DAT in train valid
do
  cat /Users/blu/artifacts/bopt/simple/exp${EXP}/42/0.0/768-buggy-init/full_lattice_log.${DAT}.json | python3 ../extract_log.py marginal unit log_prob > /Users/blu/artifacts/bopt/simple/exp${EXP}/42/0.0/768-buggy-init/lattice.${DAT}.json
  python3 ../serialize_lattice.py /Users/blu/artifacts/bopt/simple/exp${EXP}/42/0.0/768-buggy-init/lattice.${DAT}.json lattices/${DAT}/
done