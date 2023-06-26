#!/usr/bin/env bash

source vars.sh
mkdir -p ${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/v0.0.2/scripts/syn4_small/exp${EXPID}

rm -rf tmp tmp1 eval-results.json
touch tmp tmp1 eval-results.json
for SEED in 42
do
for SIZE in 768
do
for L1 in 0.01 0.1
do
for DATA in 500
do
for MIXTURE in 0.9875 0.975 0.95 0.9 0.8 0.6
do
for INPUT_NAME in dev test
do
DATA_PREFIX=${BLU_CORPORA}/vopt/syn/4/${DATA}
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/${MIXTURE}
CONFIG_NAME=${SCRIPT_PREFIX}/config${SIZE}.json
LOAD_DIR=${BLU_ARTIFACTS}/boptv2/syn4_small/exp21-1/${SEED}/${SIZE}/${L1}/${DATA}/checkpoint-final
headp ${MIXTURE} < ${DATA_PREFIX}/${INPUT_NAME}.1best.tokenizations.jsonl > tmp
tailp ${MIXTURE} --r < ${LOAD_DIR}/${INPUT_NAME}.1best.tokenizations.jsonl >> tmp\

python3 -m experiments.scripts.best_dev \
            ${OUTPUT_DIR}/log.json dev_accuracy max \
            step \
            test_accuracy \
            train_accuracy \
            dev_entroy \
            test_entropy \
            --output_json \
            --add_field seed ${SEED} \
            --add_field l1 ${L1} \
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
        ${INPUT_NAME}_boundary_f1:boundary_f1 >> eval-results.json

done
done
done
done
done
done