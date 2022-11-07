#!/usr/bin/env bash
#$ -wd /home/zlu39/jhu/bopt/scripts/ptb/exp1-14-v
#$ -V
#$ -N s3e1-14-v
#$ -j y -o /export/c01/zlu39/jobs/$JOB_NAME-$JOB_ID.out
#$ -M zlu39@jhu.edu
#$ -m e
#$ -l ram_free=35G,mem_free=35G,gpu=1,hostname=c0*|c1*
#$ -q g.q

source /home/gqin2/scripts/acquire-gpus 1
conda env list
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

EXPID="1-14-v"
mkdir -p ${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/ptb
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/ptb/exp${EXPID}
for SEED in 42
do
for SIZE in 768
do
for L1 in 0.0
do
CUDA_VISIBLE_DEVICES=1 python3 -O -um bopt.run \
    --seed ${SEED} \
    --train_dataset ${DATA_PREFIX}/ptb.train.txt \
    --eval_dataset ${DATA_PREFIX}/ptb.valid.txt \
    --input_vocab ${DATA_PREFIX}/unigram-vocab-10000.txt \
    --output_vocab ${DATA_PREFIX}/substring8-vocab-threshold=None.txt \
    --config ${SCRIPT_PREFIX}/config${SIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${L1}/${SIZE}/ \
    --overwrite_output_dir \
    --do_train --do_eval \
    --vopt \
    --bias_mode albo \
    --train_epochs 15 \
    --eval_epochs 1 \
    --save_epochs 1 \
    --train_batch_size 128 \
    --gpu_batch_size 1 \
    --l1 ${L1} \
    --task language_modeling \
    --entropic 10.0 \
    --entropy_start 13 \
    --entropy_end 14 \
    --max_blocks 8 \
    --max_block_length 32 \
    --max_unit_length 8 \
    --warmup_epochs 1 \
    --weights_learning_rate 0.0 \
    --debug_viterbi_lattice \
    --weights_file ${DATA_PREFIX}/unigram-weights-10000.txt \
    --data_num_workers 5

done
done
done
