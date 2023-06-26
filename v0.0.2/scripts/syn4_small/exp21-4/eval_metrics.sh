#!/usr/bin/env bash
source vars.sh
CUDA_VISIBLE_DEVICES=1

# this function for each run and checkpoint computes the task metrics (accuracy, and breakdown accuracies)
# and saves them to a file
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/v0.0.2/scripts/syn4_small/exp${EXPID}

rm -rf compute_metrics.tmp
for SEED in 42 44 46
do
for SIZE in 768
do
for L1 in 0.01
do
for DATA in  100 500 small full
do
for NAME in train.100 dev test
do
for LR in 0.00 6.25e-5
do
for CKPT in checkpoint-early-stopping checkpoint-final
do
for MODE in 1best lattice
do
DATA_PREFIX=${BLU_CORPORA}/vopt/syn/4/${DATA}
CKPT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/${LR}/${CKPT}
echo ${CKPT_DIR}
python3 -um experiments.scripts.accuracy \
    ${CKPT_DIR}/${NAME}.${MODE}.labels.tsv \
    ${CKPT_DIR}/${NAME}.${MODE}.predictions.tsv \
    --categories_file ${DATA_PREFIX}/${NAME}_categories.tsv \
    --output_file compute_metrics.tmp
python3 -um experiments.scripts.json_join \
    --path ${CKPT_DIR}/${NAME}.${MODE}.results.json \
    --src compute_metrics.tmp \
    --output ${CKPT_DIR}/${NAME}.${MODE}.metrics.json \
    --fields entropy
done
done
done
done
done
done
done
done
rm -rf compute_metrics.tmp