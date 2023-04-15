#!/usr/bin/env bash

for NAME in train dev test
do
python3 -m bopt.data.extract_synthetic_gold_tokenizations \
  < /export/a01/corpora/vopt/syn/4/small/${NAME}.csv \
  > /export/a01/corpora/vopt/syn/4/small/${NAME}.1best.tokenizations.jsonl
done