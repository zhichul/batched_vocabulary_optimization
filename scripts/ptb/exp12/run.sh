#!/usr/bin/env bash
EXPID="12"
mkdir -p ${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/ptb
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/ptb/exp${EXPID}
for SEED in 44
do
for SIZE in 768
do
for LR in 0.02 0.006 0.06 0.002 0.2
do
CUDA_VISIBLE_DEVICES=0 python3 -O -um bopt.run \
    --seed ${SEED} \
    --train_dataset ${DATA_PREFIX}/ptb.train.txt \
    --eval_dataset ${DATA_PREFIX}/ptb.valid.txt \
    --input_vocab ${DATA_PREFIX}/spm-unigram-vocab-10000.txt \
    --output_vocab ${DATA_PREFIX}/spm-unigram-vocab-10000.txt \
    --weights_file ${DATA_PREFIX}/spm-unigram-weights-10000.txt \
    --config ${SCRIPT_PREFIX}/config${SIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${GL}/${SIZE}/${LR} \
    --overwrite_output_dir \
    --do_train --do_eval \
    --vopt \
    --bias_mode albo \
    --train_epochs 40 \
    --eval_epochs 1 \
    --eval_steps 100  \
    --save_epochs 10 \
    --save_steps 200  \
    --train_batch_size 128 \
    --gpu_batch_size 4 \
    --task language_modeling \
    --max_blocks 6 \
    --max_block_length 32 \
    --max_unit_length 8 \
    --warmup_epochs 1 \
    --weights_learning_rate ${LR} \
    --length_normalized_initialization \
    --constant_normalization 20.91 \
    --specials "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" "[BOS]" "[EOS]" "<unk>" \
    --quiet

done
done
done
