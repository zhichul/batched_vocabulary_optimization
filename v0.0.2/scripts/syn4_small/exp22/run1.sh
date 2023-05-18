#!/usr/bin/env bash
EXPID="22"
mkdir -p ${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/v0.0.2/scripts/syn4_small/exp${EXPID}

for SEED in 42 44 46
do
for SIZE in 768
do
for VSIZE in 50 100 200 400
do
for DATA in 500 small # 100 full
do
DATA_PREFIX=${BLU_CORPORA}/vopt/syn/4/${DATA}
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${VSIZE}/${DATA}
TRAIN_NAME=train.csv
DEV_NAME=dev.csv
TEST_NAME=test.csv
CONFIG_NAME=${SCRIPT_PREFIX}/config${SIZE}.json

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
    --domain morpheme_prediction \
    --train_dataset ${DATA_PREFIX}/${TRAIN_NAME} \
    --dev_dataset ${DATA_PREFIX}/${DEV_NAME} \
    --test_dataset ${DATA_PREFIX}/${TEST_NAME} \
    --data_num_workers 1 \
    \
    --bias_mode mult_then_renorm \
    --config  ${CONFIG_NAME} \
    \
    --input_vocab ${DATA_PREFIX}/spm-unigram-vocab-${VSIZE}.txt \
    --input_tokenizer_weights ${DATA_PREFIX}/spm-unigram-weights-${VSIZE}.txt \
    --output_vocab ${DATA_PREFIX}/output_vocab.txt \
    --input_tokenizer_model unigram \
    --input_tokenizer_mode 1best \
    --special_tokens "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" \
    --pad_token "[PAD]" \
    \
    --max_blocks 1 \
    --max_unit_length 9 \
    --max_block_length 12 \
    --space_character " " \
    --remove_space \
    --split_on_space \
    \
    --task_model_learning_rate 6.25e-5 \
    --train_batch_size 1024 \
    --train_steps 1200 \
    --patience 10 \
    --lr_adjustment_window_size 2048 \
    --reduce_factor 0.25 \
    \
    --eval_steps 30 \
    \
    --gpu_batch_size 1024 \
    --device "cuda"

done
done
done
done
