#!/usr/bin/env bash
#$ -wd /home/zlu39/jhu/bopt/scripts/ptb/exp1-13
#$ -V
#$ -N s3e1-13
#$ -j y -o /export/c01/zlu39/jobs/$JOB_NAME-$JOB_ID.out
#$ -M zlu39@jhu.edu
#$ -m e
#$ -l ram_free=35G,mem_free=35G,gpu=1,hostname=c0*|c1*
#$ -q g.q

source /home/gqin2/scripts/acquire-gpus 1
conda env list
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

EXPID="1-13"
mkdir -p ${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/ptb
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/ptb/exp${EXPID}
for SEED in 42
do
for SIZE in 768
do
for WLR in 0.001 0.003 0.006 0.01 0.013 0.016 0.02 0.023 0.026 0.03 0.033 0.036 0.04 0.043 0.046 0.05
do
CUDA_VISIBLE_DEVICES=0 python3 -O -um bopt.run \
    --seed ${SEED} \
    --train_dataset ${DATA_PREFIX}/ptb.train.10k.txt \
    --eval_dataset ${DATA_PREFIX}/ptb.valid.1k.txt \
    --input_vocab ${DATA_PREFIX}/substring8-vocab-threshold=None.txt \
    --output_vocab ${DATA_PREFIX}/substring8-vocab-threshold=None.txt \
    --config ${SCRIPT_PREFIX}/config${SIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${WLR}/${SIZE}/ \
    --overwrite_output_dir --overwrite_cache \
    --do_train --do_eval \
    --vopt \
    --bias_mode albo \
    --train_epochs 5 \
    --eval_epochs 1 \
    --save_epochs 1 \
    --train_batch_size 128 \
    --gpu_batch_size 1 \
    --l1 0.0 \
    --continuing_subword_prefix @@ \
    --task language_modeling \
    --entropic 10.0 \
    --entropy_start 6 \
    --entropy_end 7 \
    --max_blocks 8 \
    --max_block_length 32 \
    --max_unit_length 8 \
    --warmup_epochs 1 \
    --weights_learning_rate ${WLR}
done
done
done
