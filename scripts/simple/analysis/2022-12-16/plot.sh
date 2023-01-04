#!/usr/bin/env bash

EXP=3-1
DIR=2022-12-16

python3 ../training_curves.py \
  $BLU_ARTIFACTS/bopt/ptb/exp${EXP}/44/0.01/768/log.json \
  --out losses.png \
  --field train_loss avg_token \
  --field_pretty train dev \
  --colors orange \
  --linestyles solid dashed \
  --names [vopt-viterbi]  \
  --xlabel "steps x" \
  --ylabel "loss" \

for GL in 0.01
do
  # extract the entropy
  cat /Users/blu/artifacts/bopt/ptb/exp${EXP}/44/${GL}/768/full_lattice_log.train.json | python3 ../extract_log.py ent > /Users/blu/artifacts/bopt/ptb/exp${EXP}/44/${GL}/768/ent.train.json
  cat /Users/blu/artifacts/bopt/ptb/exp${EXP}/44/${GL}/768/full_lattice_log.valid.json | python3 ../extract_log.py ent > /Users/blu/artifacts/bopt/ptb/exp${EXP}/44/${GL}/768/ent.valid.json

done

python3 ../training_curves.py \
  $BLU_ARTIFACTS/bopt/ptb/exp${EXP}/44/0.01/768/ent.train.json \
  $BLU_ARTIFACTS/bopt/ptb/exp${EXP}/44/0.01/768/ent.valid.json \
  --out entropy.png \
  --field ent \
  --field_pretty ent \
  --colors orange cyan \
  --linestyles solid \
  --names viterbi-train viterbi-valid \
  --xlabel "steps x100" \
  --ylabel "entropy (per sentence)"

for GL in 0.01
do
  # extract the units marginals and log_probs
  for DAT in train valid
  do
    cat /Users/blu/artifacts/bopt/ptb/exp${EXP}/44/${GL}/768/full_lattice_log.${DAT}.json | python3 ../extract_log.py marginal unit log_prob lm_marginal > /Users/blu/artifacts/bopt/ptb/exp${EXP}/44/${GL}/768/lattice.${DAT}.json
    python3 ../serialize_lattice.py /Users/blu/artifacts/bopt/ptb/exp${EXP}/44/${GL}/768/lattice.${DAT}.json lattices/${GL}/${DAT}/
  done
done