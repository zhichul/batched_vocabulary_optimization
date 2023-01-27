#!/usr/bin/env bash
EXPID="20-mult"
mkdir -p ${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/ptb
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/ptb/exp${EXPID}
for SEED in 44
do
for SIZE in 768
do
for LR in 0.02
do
for L1 in 0.1
do
CUDA_VISIBLE_DEVICES=1 python3 -O -um bopt.run \
    --seed ${SEED} \
    --train_dataset ${DATA_PREFIX}/ptb.train.txt \
    --eval_dataset ${DATA_PREFIX}/ptb.valid.txt \
    --input_vocab ${DATA_PREFIX}/substring8-vocab-threshold=None.txt \
    --output_vocab ${DATA_PREFIX}/substring8-vocab-threshold=None.txt \
    --config ${SCRIPT_PREFIX}/config${SIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${GL}/${SIZE}/${LR}/${L1} \
    --overwrite_output_dir \
    --do_train --do_eval \
    --vopt \
    --bias_mode mult_then_renorm \
    --train_epochs 25 \
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
    --continuing_subword_prefix @@ \
    --constant_normalization 20.91 \
    --l1 ${L1} \
    --specials "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" "[BOS]" "[EOS]" "<unk>" \
    --quiet

done
done
done
done
