#!/usr/bin/env bash
source vars.sh
source ../bash_scripts/inference.sh
CUDA_VISIBLE_DEVICES=1
BIAS_MODE=mult_then_renorm
inference_lattice