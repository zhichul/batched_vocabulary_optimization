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
for DATA in 100
do
for AHEAD in 100 #600
do
for TRAJ in 1
do
for WARMUP in 0
do
for BIAS in mult_then_renorm
do
for LR in 6.25e-3
do
for MOMENTUM in 0.1
do
for TRAINRANDOM in 5
do
for EVALRANDOM in 3
do
DATA_PREFIX=${BLU_CORPORA}/vopt/syn/4/${DATA}
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/${AHEAD}/${TRAJ}/${BIAS}/${WARMUP}/combined/${LR}/${MOMENTUM}/${TRAINRANDOM}/${EVALRANDOM}
for STEP in 1 15 30 45 60 75 90 105 120 135 150
do
for INPUT_NAME in train.inner train.outer dev test
do
TRAIN_INNER_NAME=train.inner.csv
TRAIN_OUTER_NAME=train.outer.csv
DEV_NAME=dev.csv
TEST_NAME=test.csv
CONFIG_NAME=${SCRIPT_PREFIX}/config${SIZE}.json
CHECKPOINT_DIR=${OUTPUT_DIR}/checkpoint-${STEP}

python3 -O -um bopt.tokenize \
      --input_vocab ${CHECKPOINT_DIR}/learned_vocab.txt \
      --input_tokenizer_weights ${CHECKPOINT_DIR}/learned_vocab.txt \
      --input_tokenizer_model unigram \
      --input_tokenizer_mode 1best \
      --special_tokens "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" \
      --pad_token "[PAD]" \
      \
      --max_blocks 1 \
      --max_unit_length 9 \
      --max_block_length 12 \
      --space_character " " \
      --report_reference \
      --input_mode json \
      < ${DATA_PREFIX}/${INPUT_NAME}.1best.tokenizations.jsonl \
      > ${CHECKPOINT_DIR}/${INPUT_NAME}.1best.tokenizations.jsonl

      python3 -um bopt.tokenization.evaluate_tokenization \
      ${DATA_PREFIX}/${INPUT_NAME}.1best.tokenizations.jsonl \
      ${CHECKPOINT_DIR}/${INPUT_NAME}.1best.tokenizations.jsonl \
      --report_reference \
      --skip_inf \
      --categories_file ${DATA_PREFIX}/${INPUT_NAME}.tokenization_categories.jsonl > ${CHECKPOINT_DIR}/${INPUT_NAME}.1best.tokenizations.f1.json \

      echo ${CHECKPOINT_DIR}/${INPUT_NAME}.1best.tokenizations.f1.json
      cat ${CHECKPOINT_DIR}/${INPUT_NAME}.1best.tokenizations.f1.json

done
done
done
done
done
done
done
done
done
done
done
done
done
done