#!/usr/bin/env bash
source vars.sh
mkdir -p ${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/superbizarre/data
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/v0.0.2/scripts/superbizarre/exp${EXPID}

for SEED in 42 44 46
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/
TRAIN_NAME=${DOMAIN}/csv/${DOMAIN}_train.csv
DEV_NAME=${DOMAIN}/csv/${DOMAIN}_dev.csv
TEST_NAME=${DOMAIN}/csv/${DOMAIN}_test.csv
PRETRAINED_MODEL=${BLU_ARTIFACTS}/bert
CONFIG_NAME=${SCRIPT_PREFIX}/config768.json

CUDA_VISIBLE_DEVICES=0 python3 -O -um bopt.train \
    --output_directory ${OUTPUT_DIR} \
    --overwrite_output_directory \
    \
    --train_tokenization_cache $OUTPUT_DIR/cache/train-${TRAIN_NAME}-tokenization \
    --dev_tokenization_cache $OUTPUT_DIR/cache/dev-${DEV_NAME}-tokenization \
    --test_tokenization_cache $OUTPUT_DIR/cache/test-${TEST_NAME}-tokenization \
    --overwrite_cache \
    \
    --seed ${SEED} \
    --task classification \
    --domain superbizarre_prediction_gold \
    --train_dataset ${DATA_PREFIX}/${TRAIN_NAME} \
    --dev_dataset ${DATA_PREFIX}/${DEV_NAME} \
    --test_dataset ${DATA_PREFIX}/${TEST_NAME} \
    --data_num_workers 1 \
    \
    --bias_mode mult_then_renorm \
    --config ${CONFIG_NAME} \
    --pretrained_model ${PRETRAINED_MODEL} \
    --pretrained_ignore cls.predictions.decoder.weight cls.predictions.bias \
    \
    --input_vocab ${BLU_ARTIFACTS}/bert/vocab.txt \
    --input_tokenizer_weights ${BLU_ARTIFACTS}/bert/tokenizer.json  \
    --output_vocab ${DATA_PREFIX}/labels.txt \
    --input_tokenizer_model bert \
    --input_tokenizer_mode bert \
    --special_tokens "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" \
    --pad_token "[PAD]" \
    \
    --task_model_learning_rate 6.25e-5 \
    --train_batch_size 1024 \
    --train_steps 1200 \
    --patience 10 \
    --lr_adjustment_window_size 58447 \
    --reduce_factor 0.25 \
    \
    --eval_steps 60 \
    \
    --gpu_batch_size 128 \
    --device "cuda" \
    \
    --gold_percentage 1.0 \

done