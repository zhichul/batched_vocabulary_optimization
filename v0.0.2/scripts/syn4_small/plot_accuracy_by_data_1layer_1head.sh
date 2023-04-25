#!/usr/bin/env bash

python3 -m experiments.plotting.bar \
  --output accuracy_by_data_1layer_1head.png \
  --bar exp22-2/hyperopt-results.json UnigramLM data test_accuracy gray \
  --bar exp23-2/hyperopt-results.json OpTok data test_accuracy orange \
  --bar exp24-2/hyperopt-results.json E2E-ALBO data test_accuracy green \
  --bar exp21-2/hyperopt-results.json E2E data test_accuracy blue \
  --xmap full 11k \
  --xmap small 2k \
  --width 0.9 \
  --ylim 0.5 1.0 \
  --xlabel "number of training words" \
  --ylabel "test accuracy" \
