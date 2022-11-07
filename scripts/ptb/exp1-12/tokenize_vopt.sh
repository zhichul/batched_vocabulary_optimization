#!/usr/bin/env bash
#$ -wd /home/zlu39/jhu/bopt/scripts/ptb/exp1-12
#$ -V
#$ -N s3e1-12
#$ -j y -o /export/c01/zlu39/jobs/$JOB_NAME-$JOB_ID.out
#$ -M zlu39@jhu.edu
#$ -m e
#$ -l ram_free=35G,mem_free=35G,gpu=1,hostname=c0*|c1*
#$ -q g.q

source /home/gqin2/scripts/acquire-gpus 1
conda env list
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

EXPID="1-12"
mkdir -p ${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/ptb
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/ptb/exp${EXPID}
for SEED in 42
do
for SIZE in 768
do
for L1 in 0.0 0.1
do
for CKPT in 100 1000 2000 3000
do
CUDA_VISIBLE_DEVICES=1 python3 -O -um bopt.run \
    --do_tokenize \
    --seed ${SEED} \
    --eval_dataset ${DATA_PREFIX}/ptb.valid.txt \
    --input_vocab ${DATA_PREFIX}/substring8-vocab-threshold=None.txt \
    --weights_file ${ARTIFACT_PREFIX}/${SEED}/${L1}/${SIZE}/checkpoint-${CKPT}/learned_vocab.txt \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${L1}/${SIZE}/ \
    --overwrite_output_dir \
    --config ${SCRIPT_PREFIX}/config${SIZE}.json \
    --continuing_subword_prefix @@ \
    --task language_modeling > ${ARTIFACT_PREFIX}/${SEED}/${L1}/${SIZE}/vopt-${CKPT}.seg.txt
done
done
done
done