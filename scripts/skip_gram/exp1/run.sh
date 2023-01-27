#!/usr/bin/env bash
EXPID="1"
mkdir -p ${BLU_ARTIFACTS}/bopt/skip_gram/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/ptb
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/skip_gram/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/skip_gram/exp${EXPID}
for SEED in 44
do
for SIZE in 768
do
for LR in 0.02
do
CUDA_VISIBLE_DEVICES=0 python3 -O -um bopt.run \
    --seed ${SEED} \
    --train_dataset ${DATA_PREFIX}/ptb.train.txt \
    --eval_dataset ${DATA_PREFIX}/ptb.valid.txt \
    --input_vocab ${DATA_PREFIX}/substring8-vocab-threshold=None.txt \
    --output_vocab ${DATA_PREFIX}/substring8-vocab-threshold=None.txt \
    --config ${SCRIPT_PREFIX}/config${SIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${LR} \
    --overwrite_output_dir --overwrite_cache \
    --do_train --do_eval \
    --vopt \
    --bias_mode albo \
    --train_epochs 60 \
    --eval_epochs 1 \
    --eval_steps 2500  \
    --save_epochs 10 \
    --save_steps 2500  \
    --train_batch_size 1536 \
    --gpu_batch_size 48 \
    --task skip_gram \
    --max_blocks 2 \
    --max_block_length 20 \
    --max_unit_length 8 \
    --warmup_epochs 1 \
    --weights_learning_rate ${LR} \
    --length_normalized_initialization \
    --continuing_subword_prefix @@ \
    --no_normalization \
    --specials "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" "[BOS]" "[EOS]" "<unk>" \
    --skip_gram_distances 1 2 3 4 5 6 7 8 9 10 \
    --data_num_workers 1

done
done
done
