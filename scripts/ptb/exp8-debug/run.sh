#!/usr/bin/env bash
#$ -wd /home/zlu39/jhu/bopt/scripts/ptb/exp8-debug
#$ -V
#$ -N s3e8-debug
#$ -j y -o /export/c01/zlu39/jobs/$JOB_NAME-$JOB_ID.out
#$ -M zlu39@jhu.edu
#$ -m e
#$ -l ram_free=35G,mem_free=35G,gpu=1,hostname=c0*|c1*
#$ -q g.q

source /home/gqin2/scripts/acquire-gpus 1
conda env list
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

EXPID="8-debug"
mkdir -p ${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/ptb
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/ptb/exp${EXPID}
for SEED in 44
do
for SIZE in 768
do
CUDA_VISIBLE_DEVICES=0 python3 -O -um bopt.run \
    --seed ${SEED} \
    --train_dataset ${DATA_PREFIX}/ptb.valid.128.txt \
    --eval_dataset ${DATA_PREFIX}/ptb.valid.128.txt \
    --input_vocab ${DATA_PREFIX}/spm-unigram-vocab-10000.txt \
    --output_vocab ${DATA_PREFIX}/spm-unigram-vocab-10000.txt \
    --weights_file ${DATA_PREFIX}/spm-unigram-weights-10000.txt \
    --config ${SCRIPT_PREFIX}/config${SIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${SIZE}/ \
    --overwrite_output_dir \
    --do_train --do_eval \
    --vopt \
    --bias_mode albo \
    --train_epochs 125 \
    --eval_epochs 1 \
    --save_epochs 1 \
    --train_batch_size 128 \
    --gpu_batch_size 4 \
    --task language_modeling \
    --max_blocks 6 \
    --max_block_length 32 \
    --max_unit_length 8 \
    --warmup_epochs 1 \
    --weights_learning_rate 0.0 \
    --normalize_by_tokens \
    --debug_viterbi_lattice \
    --specials "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" "[BOS]" "[EOS]" "<unk>" \


done
done
