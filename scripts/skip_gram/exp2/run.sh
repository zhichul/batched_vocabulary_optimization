#!/usr/bin/env bash

EXPID="2"
mkdir -p ${BLU_ARTIFACTS}/bopt/skip_gram/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/ptb
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/skip_gram/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/skip_gram/exp${EXPID}
for SEED in 44 # 42 46
do
for SIZE in 768
do
for VSIZE in 10000 #  8000 6000 4000 2000
do
CUDA_VISIBLE_DEVICES=1 python3 -O -um bopt.run \
    --seed ${SEED} \
    --train_dataset ${DATA_PREFIX}/ptb.train.txt \
    --eval_dataset ${DATA_PREFIX}/ptb.valid.txt \
    --input_vocab ${DATA_PREFIX}/spm-unigram-vocab-${VSIZE}.txt \
    --weights_file ${DATA_PREFIX}/spm-unigram-weights-${VSIZE}.txt \
    --output_vocab ${DATA_PREFIX}/spm-unigram-vocab-${VSIZE}.txt  \
    --config ${SCRIPT_PREFIX}/config${SIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${VSIZE} \
    --overwrite_output_dir --overwrite_cache \
    --do_train \
    --train_epochs 15 \
    --eval_steps 5000 \
    --save_steps 1000000 \
    --eval_epochs 1 \
    --save_epochs 5 \
    --train_batch_size 1536 \
    --gpu_batch_size 512 \
    --task skip_gram \
    --warmup_epochs 1 \
    --data_num_workers 0 \
    --max_blocks 2 \
    --max_block_length 20 \
    --max_unit_length 8 \
    --max_length 20 \
    --specials "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" "[BOS]" "[EOS]" "<unk>" \
    --skip_gram_distances 1 2 3 4 5 6 7 8 9 10 \
    --data_num_workers 6 \
    --quiet \


done
done
done
