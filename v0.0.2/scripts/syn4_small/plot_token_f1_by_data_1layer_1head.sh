#!/usr/bin/env bash

python3 -m experiments.plotting.bar \
  --output token_f1_by_data_1layer_1head.png \
  --bar exp22-2/hyperopt-results.json UnigramLM data token_f1 gray \
  --bar exp23-2/hyperopt-results.json OpTok data token_f1 orange \
  --bar exp24-2/hyperopt-results.json E2E-ALBO data token_f1 green \
  --bar exp21-2/hyperopt-results.json E2E data token_f1 blue \
  --xmap full 11k \
  --xmap small 2k \
  --width 0.9 \
  --ylim 0.3 1.0 \
  --xlabel "number of training words" \
  --ylabel "boundary f1" \
