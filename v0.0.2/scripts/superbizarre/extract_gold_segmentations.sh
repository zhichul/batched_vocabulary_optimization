#!/usr/bin/env bash

for DOMAIN in arxiv amazon reddit
do
for NAME in train dev test
do
tail -n +2 /export/a01/corpora/superbizarre/data/${DOMAIN}/csv/${DOMAIN}_${NAME}.csv \
  | csv_extractor 0 \
  | python3 -m bopt.superbizarre.derivator \
  > /export/a01/corpora/superbizarre/data/${DOMAIN}/csv/${DOMAIN}_${NAME}.1best.tokenizations.jsonl
tail -n +2 /export/a01/corpora/superbizarre/data/${DOMAIN}/csv/${DOMAIN}_${NAME}.csv \
  | csv_extractor 0 \
  | python3 -m bopt.superbizarre.categories \
  > /export/a01/corpora/superbizarre/data/${DOMAIN}/csv/${DOMAIN}_${NAME}.tokenization_categories.jsonl
done
done