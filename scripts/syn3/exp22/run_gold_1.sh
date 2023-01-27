#!/usr/bin/env bash

EXPID="22"
mkdir -p ${BLU_ARTIFACTS}/bopt/syn3/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/vopt/syn/3/full
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/syn3/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/syn3/exp${EXPID}
for SEED in 42
do
for LAYER in 1
do
for HEAD in 1 2 4
do
for SIZE in 24 96 192
do
for L1 in gold_test
do
CUDA_VISIBLE_DEVICES=1 python3 -O -um bopt.run \
    --seed ${SEED} \
    --train_dataset ${DATA_PREFIX}/train.csv \
    --eval_dataset ${DATA_PREFIX}/dev.csv \
    --test_dataset ${DATA_PREFIX}/test.csv \
    --input_vocab ${DATA_PREFIX}/substring-vocab-threshold=None.txt \
    --weights_file /export/a01/artifacts/bopt/syn3/exp21/42/1/1/24/0.01/checkpoint-10000/learned_vocab.txt  \
    --output_vocab ${DATA_PREFIX}/output_vocab.txt \
    --config ${SCRIPT_PREFIX}/config${LAYER}-${HEAD}-${SIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${LAYER}/${HEAD}/${SIZE}/${L1}/ \
    --overwrite_output_dir \
    --overwrite_cache \
    --do_eval --do_train \
    --train_epochs 3000 \
    --eval_steps 20 \
    --save_steps 10000000 \
    --save_epochs 1500 \
    --train_batch_size 1024 \
    --gpu_batch_size 1024 \
    --task morpheme_prediction \
    --max_blocks 1 \
    --max_block_length 12 \
    --max_unit_length 9 \
    --specials "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" \
    --continuing_subword_prefix @@ \
    --max_length 12 \
    --quiet \
    --segmentation_dictionary ${DATA_PREFIX}/train.csv ${DATA_PREFIX}/dev.csv ${DATA_PREFIX}/test.csv 

done
done
done
done
done
