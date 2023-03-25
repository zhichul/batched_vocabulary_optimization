#!/usr/bin/env bash
EXPID="r1-2"
mkdir -p ${BLU_ARTIFACTS}/bopt/snli_1.0/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/snli_1.0
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/snli_1.0/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/snli_1.0/exp${EXPID}
for SEED in 42
do
for SIZE in "2-4-192" # 768
do
for L1 in 0.1 # 0.01 1.0
do
for VSIZE in  16000 # 24000
do
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3 -O -um bopt.run \
    --seed ${SEED} \
    --train_dataset "${DATA_PREFIX}/snli_1.0_train.length<150.first100k.csv" \
    --eval_dataset "${DATA_PREFIX}/snli_1.0_dev.length<150.csv" \
    --test_dataset "${DATA_PREFIX}/snli_1.0_test.length<150.csv" \
    --input_vocab ${DATA_PREFIX}/spm-unigram-vocab-${VSIZE}.txt \
    --weights_file ${DATA_PREFIX}/spm-unigram-weights-${VSIZE}.txt  \
    --output_vocab ${DATA_PREFIX}/output_vocab.txt \
    --config ${SCRIPT_PREFIX}/config${SIZE}-${VSIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${L1}/${SIZE}/${VSIZE} \
    --overwrite_output_dir \
    --do_train --do_eval \
    --train_epochs 10 \
    --eval_steps 100 \
    --save_steps 10000000 \
    --save_epochs 1 \
    --train_batch_size 1024 \
    --gpu_batch_size 64 \
    --task sentiment_analysis \
    --max_blocks 1 \
    --max_block_length 160 \
    --max_unit_length 8 \
    --max_length 160 \
    --specials "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" \
    --quiet \
    --overwrite_cache \

done
done
done
done
