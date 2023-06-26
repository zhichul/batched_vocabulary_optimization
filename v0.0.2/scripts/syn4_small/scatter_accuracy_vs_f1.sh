#!/usr/bin/env bash

python3 -m experiments.plotting.scatter \
        --output test_accuracy_vs_test_boundary_f1.pdf \
        --scatter exp22-gold-mixture-raw/eval-results-test.json "Reference + Noise" "test_boundary_f1" "test_accuracy" orange o \
        --scatter exp21-1-random/extract.500.json "E2E 1B-RD" "test_boundary_f1" "test_accuracy"  violet o \
        --scatter exp22/extract.500.json "UnigramLM" "test_boundary_f1" "test_accuracy"  dodgerblue o \
        --xlim 0.3 1.0  \
        --ylim 0.6 1.0  \
        --xlabel "Tokenization Boundary F1" \
        --ylabel "Morpheme Prediction Accuracy"