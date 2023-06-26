EXPID="1-4"
DATA_PREFIX=${BLU_CORPORA}/superbizarre/data
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/v0.0.2/scripts/superbizarre/exp${EXPID}

for DOMAIN in amazon
do
for INPUT_NAME in dev test
do
for CKPT in checkpoint-early-stopping checkpoint-final
do
for L1 in 0.01 0.1 1.0
do
for SEED in 42 44 46
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${L1}
CHECKPOINT_DIR=${OUTPUT_DIR}/${CKPT}

CUDA_VISIBLE_DEVICES=0 python3 -O -um bopt.tokenize \
    --input_vocab ${CHECKPOINT_DIR}/learned_vocab.txt \
    --input_tokenizer_weights ${CHECKPOINT_DIR}/learned_vocab.txt \
    --input_tokenizer_model unigram \
    --input_tokenizer_mode 1best \
    --special_tokens "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" \
    --pad_token "[PAD]" \
    \
    --max_blocks 1 \
    --max_unit_length 19 \
    --max_block_length 48 \
    --space_character "▁" \
    --report_reference \
    --input_mode json \
    < ${DATA_PREFIX}/${DOMAIN}/csv/${DOMAIN}_${INPUT_NAME}.1best.tokenizations.jsonl \
    > ${CHECKPOINT_DIR}/${DOMAIN}_${INPUT_NAME}.1best.tokenizations.jsonl

done
done
done
done
done
