#!/usr/bin/env bash
EXPID="21-3"
mkdir -p ${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/vopt/syn/4/small
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/v0.0.2/scripts/syn4_small/exp${EXPID}

for DATA in 100 500 small full
do
for SIZE in 768
do
for INPUT_NAME in train dev test
do
for CKPT in checkpoint-early-stopping checkpoint-final
do
for SEED in 42 44 46 48 50 52 54 56 58 60
do
for L1 in 0.01 0.1 1.0
do
for CKPT_LOAD in checkpoint-early-stopping checkpoint-final
do
for POS in token-pos char-pos
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/${CKPT_LOAD}/${POS}
CHECKPOINT_DIR=${OUTPUT_DIR}/${CKPT}

python3 -um bopt.tokenization.evaluate_tokenization \
    ${DATA_PREFIX}/${INPUT_NAME}.1best.tokenizations.jsonl \
    ${CHECKPOINT_DIR}/${INPUT_NAME}.1best.tokenizations.jsonl > ${CHECKPOINT_DIR}/${INPUT_NAME}.1best.tokenizations.f1.json

echo ${CHECKPOINT_DIR}/${INPUT_NAME}.1best.tokenizations.f1.json
cat ${CHECKPOINT_DIR}/${INPUT_NAME}.1best.tokenizations.f1.json

done
done
done
done
done
echo ""
done
echo ""
done
echo ""
done
