#!/usr/bin/env bash
source vars.sh
source ../bash_scripts/inference.sh
CUDA_VISIBLE_DEVICES=0
BIAS_MODE=albo
inference_lattice
