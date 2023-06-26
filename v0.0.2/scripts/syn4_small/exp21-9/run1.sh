#!/usr/bin/env bash
source vars.sh
mkdir -p ${BLU_ARTIFACTS2}/boptv2/syn4_small/exp${EXPID}
ARTIFACT_PREFIX=${BLU_ARTIFACTS2}/boptv2/syn4_small/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/v0.0.2/scripts/syn4_small/exp${EXPID}

for SEED in 42
do
for SIZE in 768
do
for L1 in 0.00
do
for DATA in 500
do
for AHEAD in 18
do
for TRAJ in 1
do
for WARMUP in 50 100
do
for BIAS in mult_then_renorm
do
for LR in 6.25e-5
do
DATA_PREFIX=${BLU_CORPORA}/vopt/syn/4/${DATA}
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/${AHEAD}/${TRAJ}/${BIAS}/${WARMUP}/combined/${LR}
TRAIN_INNER_NAME=train.inner.csv
TRAIN_OUTER_NAME=train.outer.csv
DEV_NAME=dev.csv
TEST_NAME=test.csv
CONFIG_NAME=${SCRIPT_PREFIX}/config${SIZE}.json

CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python3 -O -um bopt.train_bilevel \
    --output_directory ${OUTPUT_DIR} \
    --overwrite_output_directory \
    \
    --train_inner_tokenization_cache $OUTPUT_DIR/cache/train-${TRAIN_INNER_NAME}-tokenization \
    --train_outer_tokenization_cache $OUTPUT_DIR/cache/train-${TRAIN_OUTER_NAME}-tokenization \
    --dev_tokenization_cache $OUTPUT_DIR/cache/dev-${DEV_NAME}-tokenization \
    --test_tokenization_cache $OUTPUT_DIR/cache/test-${TEST_NAME}-tokenization \
    --overwrite_cache \
    \
    --seed ${SEED} \
    --task classification \
    --domain morpheme_prediction \
    --train_inner_dataset ${DATA_PREFIX}/${TRAIN_INNER_NAME} \
    --train_outer_dataset ${DATA_PREFIX}/${TRAIN_OUTER_NAME} \
    --dev_dataset ${DATA_PREFIX}/${DEV_NAME} \
    --test_dataset ${DATA_PREFIX}/${TEST_NAME} \
    --data_num_workers 1 \
    \
    --bias_mode ${BIAS} \
    --config  ${CONFIG_NAME} \
    \
    --input_vocab ${DATA_PREFIX}/substring-vocab-max_length=9-min_count=1.txt \
    --output_vocab ${DATA_PREFIX}/output_vocab.txt \
    --input_tokenizer_model unigram \
    --input_tokenizer_mode lattice \
    --special_tokens "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" \
    --pad_token "[PAD]" \
    --log_space_parametrization \
    \
    --max_blocks 1 \
    --max_unit_length 9 \
    --max_block_length 12 \
    --space_character " " \
    --remove_space \
    --split_on_space \
    \
    --task_model_learning_rate ${LR} \
    --input_tokenizer_learning_rate 0.02 \
    --train_batch_size 1024 \
    --train_steps 100 \
    --patience 10 \
    --lr_adjustment_window_size 2048 \
    --reduce_factor 0.25 \
    \
    --eval_steps 1 \
    \
    --annealing 10.0 \
    --annealing_start_steps 300 \
    --annealing_end_steps 450 \
    --L1 ${L1} \
    \
    --gpu_batch_size 32 \
    --device "cuda" \
    \
    --train_steps_inner ${AHEAD} \
    --train_trajectory_inner ${TRAJ} \
    --train_steps_warmup ${WARMUP} \
    --inner_threshold 1e-3 \
    --bilevel_optimization_scheme unroll \
    --train_batch_size_inner 64 \
    --gpu_batch_size_inner 64 \
    --inner_optimizer Adam \
    --random_restarts 3 \
    --eval_random_restarts 2 \

done
done
done
done
done
done
done
done
done
