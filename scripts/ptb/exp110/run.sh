#!/usr/bin/env bash

EXPID="110"
mkdir -p ${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/ptb
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/ptb/exp${EXPID}
for SEED in 44 # 42 46
do
for SIZE in 768
do
for VSIZE in 10000 8000 6000
do
CUDA_VISIBLE_DEVICES=0 python3 -O -um bopt.run \
    --seed ${SEED} \
    --train_dataset ${DATA_PREFIX}/ptb.train.txt \
    --eval_dataset ${DATA_PREFIX}/ptb.valid.txt \
    --test_dataset ${DATA_PREFIX}/ptb.test.txt \
    --input_vocab ${DATA_PREFIX}/spm-unigram-vocab-${VSIZE}.txt \
    --weights_file ${DATA_PREFIX}/spm-unigram-weights-${VSIZE}.txt \
    --output_vocab ${DATA_PREFIX}/spm-unigram-vocab-${VSIZE}.txt  \
    --config ${SCRIPT_PREFIX}/config${SIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${VSIZE} \
    --overwrite_output_dir --overwrite_cache \
    --do_train \
    --train_epochs 30 \
    --eval_epochs 1 \
    --save_epochs 10 \
    --train_batch_size 128 \
    --gpu_batch_size 64 \
    --task language_modeling \
    --warmup_epochs 1 \
    --data_num_workers 0 \
    --max_blocks 6 \
    --max_block_length 32 \
    --max_unit_length 8 \
    --max_length 128 \
    --specials "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" "[BOS]" "[EOS]" "<unk>" \
    --quiet \


done
done
done
