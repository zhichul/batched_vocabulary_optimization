#!/usr/bin/env bash
#$ -wd /home/zlu39/jhu/bopt/scripts/simple/exp1-21
#$ -V
#$ -N s3e1-21
#$ -j y -o /export/c01/zlu39/jobs/$JOB_NAME-$JOB_ID.out
#$ -M zlu39@jhu.edu
#$ -m e
#$ -l ram_free=35G,mem_free=35G,gpu=1,hostname=c0*|c1*
#$ -q g.q

source /home/gqin2/scripts/acquire-gpus 1
conda env list
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

EXPID="1-21"
mkdir -p ${BLU_ARTIFACTS}/bopt/simple/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/simple
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/simple/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/simple/exp${EXPID}
for SEED in 42
do
for SIZE in 768
do
for GL in 0.0
do
CUDA_VISIBLE_DEVICES=0 python3 -O -um bopt.run \
--seed ${SEED} \
    --train_dataset ${DATA_PREFIX}/se.train.txt \
    --eval_dataset ${DATA_PREFIX}/se.valid.txt \
    --input_vocab ${DATA_PREFIX}/unigram-vocab-10000.txt \
    --output_vocab ${DATA_PREFIX}/unigram-vocab-10000.txt \
    --weights_file ${DATA_PREFIX}/unigram-weights-10000.txt \
    --config ${SCRIPT_PREFIX}/config${SIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${GL}/${SIZE}/ \
    --overwrite_output_dir \
    --do_train --do_eval \
    --vopt \
    --bias_mode albo \
    --train_epochs 75 \
    --eval_epochs 1 \
    --save_epochs 1 \
    --train_batch_size 128 \
    --gpu_batch_size 4 \
    --task language_modeling \
    --entropic 0.0 \
    --entropy_start -1 \
    --entropy_end 0 \
    --entropy_start_dec 15 \
    --entropy_end_dec 45 \
    --max_blocks 4 \
    --max_block_length 32 \
    --max_unit_length 8 \
    --warmup_epochs 1 \
    --length_normalized_initialization \
    --weights_learning_rate 0.0 \
    --debug_fixed_point \
    --group_lasso ${GL} \
    --output_viterbi \
    --normalize_by_tokens

done
done
done
