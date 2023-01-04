#!/usr/bin/env bash
#$ -wd /home/zlu39/jhu/bopt/scripts/ptb/exp4-1
#$ -V
#$ -N ptbe4-1
#$ -j y -o /export/c01/zlu39/jobs/$JOB_NAME-$JOB_ID.out
#$ -M zlu39@jhu.edu
#$ -m e
#$ -l ram_free=35G,mem_free=35G,gpu=1,hostname=c0*|c1*
#$ -q g.q

source /home/gqin2/scripts/acquire-gpus 1
conda env list
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

EXPID="4-1"
mkdir -p ${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/ptb
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/ptb/exp${EXPID}
for SEED in 42 44 46
do
for SIZE in 768
do
CUDA_VISIBLE_DEVICES=0 python3 -O -um bopt.run \
    --seed ${SEED} \
    --train_dataset ${DATA_PREFIX}/ptb.train.txt \
    --eval_dataset ${DATA_PREFIX}/ptb.valid.txt \
    --input_vocab ${DATA_PREFIX}/unigram-vocab-10000.txt \
    --weights_file ${DATA_PREFIX}/unigram-weights-10000.txt \
    --output_vocab ${DATA_PREFIX}/unigram-vocab-10000.txt  \
    --config ${SCRIPT_PREFIX}/config${SIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${SIZE}/ \
    --overwrite_output_dir --overwrite_cache \
    --do_train --do_eval \
    --train_epochs 75 \
    --eval_epochs 1 \
    --save_epochs 10 \
    --train_batch_size 128 \
    --gpu_batch_size 64 \
    --task language_modeling \
    --max_length 256 \
    --max_unit_length 8 \
    --warmup_epochs 1 \
    --data_num_workers 0 \
    --debug_node_unigram \
    --quiet


done
done
