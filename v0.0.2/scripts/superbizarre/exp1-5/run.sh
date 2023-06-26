#!/usr/bin/env bash
EXPID="1-5"
mkdir -p ${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/superbizarre/data
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/v0.0.2/scripts/superbizarre/exp${EXPID}

for SEED in 42 44 46
do
for L1 in 0.1 1.0 0.01
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${L1}
TRAIN_NAME=amazon/csv/amazon_train.csv
DEV_NAME=amazon/csv/amazon_dev.csv
TEST_NAME=amazon/csv/amazon_test.csv
PRETRAINED_MODEL=${BLU_ARTIFACTS}/bert
CONFIG_NAME=${SCRIPT_PREFIX}/config768.json

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3 -O -um bopt.train \
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
    --domain superbizarre_prediction \
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
    --output_vocab ${DATA_PREFIX}/labels.txt \
    --input_tokenizer_model unigram \
    --input_tokenizer_mode lattice \
    --special_tokens "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" \
    --pad_token "[PAD]" \
    \
    --max_blocks 1 \
    --max_unit_length 19 \
    --max_block_length 48 \
    --space_character "▁" \
    --split_on_space \
    --collapse_padding \
    \
    --task_model_learning_rate 6.25e-5 \
    --input_tokenizer_learning_rate 0.02 \
    --train_batch_size 1024 \
    --train_steps 1200 \
    --patience 10 \
    --lr_adjustment_window_size 58447 \
    --reduce_factor 0.25 \
    \
    --eval_steps 60 \
    \
    --annealing 10.0 \
    --annealing_start_steps 600 \
    --annealing_end_steps 900 \
    --L1 ${L1} \
    \
    --gpu_batch_size 128 \
    --device "cuda"
done
done