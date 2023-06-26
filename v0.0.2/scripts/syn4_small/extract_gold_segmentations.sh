#!/usr/bin/env bash

for DATA in 100 500 small full
do
for NAME in train.inner train.outer #train.100 dev test
do
python3 -m bopt.data.extract_synthetic_gold_tokenizations \
  < /export/a01/corpora/vopt/syn/4/${DATA}/${NAME}.csv \
  > /export/a01/corpora/vopt/syn/4/${DATA}/${NAME}.1best.tokenizations.jsonl
python3 -m bopt.data.extract_synthetic_tokenization_categories \
  < /export/a01/corpora/vopt/syn/4/${DATA}/${NAME}.csv \
  > /export/a01/corpora/vopt/syn/4/${DATA}/${NAME}.tokenization_categories.jsonl
done
done