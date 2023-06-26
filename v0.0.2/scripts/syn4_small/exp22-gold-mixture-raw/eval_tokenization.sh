#!/usr/bin/env bash

source vars.sh
mkdir -p ${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/v0.0.2/scripts/syn4_small/exp${EXPID}

for INPUT_NAME in dev test
do
rm -rf tmp tmp1 eval-results-${INPUT_NAME}.json
touch tmp tmp1 eval-results-${INPUT_NAME}.json
for SEED in 42
do
for SIZE in 768
do
for DATA in 500
do
for MIXTURE in 0.0 # 0.9875 0.975 0.95 0.9 0.8 0.6 0.4 0.2 0.0
do
DATA_PREFIX=${BLU_CORPORA}/vopt/syn/4/${DATA}
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${DATA}/${MIXTURE}
TRAIN_NAME=train.csv
DEV_NAME=dev.csv
TEST_NAME=test.csv
CONFIG_NAME=${SCRIPT_PREFIX}/config${SIZE}.json

for INPUT_NAME in dev test
do
python3 -O -um bopt.tokenize \
    --input_vocab ${OUTPUT_DIR}/checkpoint-final/learned_vocab.txt \
    --input_tokenizer_weights ${OUTPUT_DIR}/checkpoint-final/learned_vocab.txt \
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
    > ${OUTPUT_DIR}/checkpoint-final/${INPUT_NAME}.1best.tokenizations.jsonl
done
exit

headp ${MIXTURE} < ${DATA_PREFIX}/${INPUT_NAME}.1best.tokenizations.jsonl > tmp
tailp ${MIXTURE} --r < ${OUTPUT_DIR}/checkpoint-final/${INPUT_NAME}.1best.tokenizations.jsonl >> tmp \

python3 -m experiments.scripts.best_dev \
            ${OUTPUT_DIR}/log.json dev_accuracy max \
            step \
            test_accuracy \
            train_accuracy \
            dev_entroy \
            test_entropy \
            --output_json \
            --add_field seed ${SEED} \
            --add_field mixture ${MIXTURE} > tmp1

python3 -um bopt.tokenization.evaluate_tokenization \
      ${DATA_PREFIX}/${INPUT_NAME}.1best.tokenizations.jsonl \
      tmp > $OUTPUT_DIR/${INPUT_NAME}.1best.tokenizations.f1.json \

python3 -m experiments.scripts.json_join \
        --path $OUTPUT_DIR/${INPUT_NAME}.1best.tokenizations.f1.json \
        --src tmp1 \
        --fields \
        ${INPUT_NAME}_boundary_precision:boundary_precision \
        ${INPUT_NAME}_boundary_recall:boundary_recall \
        ${INPUT_NAME}_boundary_f1:boundary_f1 >> eval-results-${INPUT_NAME}.json

done
done
done
done
python3 -m experiments.plotting.scatter \
        --output ${INPUT_NAME}_accuracy_vs_${INPUT_NAME}_boundary_f1.png \
        --scatter eval-results-${INPUT_NAME}.json "mixture of tokenizers" "${INPUT_NAME}_boundary_f1" "${INPUT_NAME}_accuracy" ${METRIC_Y[${METRIC}]}  teal o \
        --xlim 0.4 1.0  \
        --ylim 0.7 1.0  \
        --xlabel "tokenization f1" \
        --ylabel "task accuracy"
done

