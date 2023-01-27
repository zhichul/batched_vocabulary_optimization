#!/usr/bin/env bash
#$ -wd /home/zlu39/jhu/bopt/scripts/syn3/exp34-11
#$ -V
#$ -N s3e34-11
#$ -j y -o /export/c01/zlu39/jobs/$JOB_NAME-$JOB_ID.out
#$ -M zlu39@jhu.edu
#$ -m e
#$ -l ram_free=35G,mem_free=35G,gpu=1,hostname=c0*|c1*
#$ -q g.q

source /home/gqin2/scripts/acquire-gpus 1
conda env list
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

EXPID="64-11"
mkdir -p ${BLU_ARTIFACTS}/bopt/syn3/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/vopt/syn/3/full
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/syn3/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/syn3/exp${EXPID}

for SEED in 42
do
for LAYER in 2
do
for HEAD in 1 2 4
do
for SIZE in 24 96 192
do
for L1 in 0.1
do
CUDA_VISIBLE_DEVICES=1 python3 -O -um bopt.run \
    --seed ${SEED} \
    --train_dataset ${DATA_PREFIX}/train.csv \
    --eval_dataset ${DATA_PREFIX}/dev.csv \
    --test_dataset ${DATA_PREFIX}/test.csv \
    --input_vocab ${DATA_PREFIX}/substring-vocab-threshold=None.txt \
    --output_vocab ${DATA_PREFIX}/output_vocab.txt \
    --config ${SCRIPT_PREFIX}/config${LAYER}-${HEAD}-${SIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${LAYER}/${HEAD}/${SIZE}/${L1}/ \
    --overwrite_output_dir --overwrite_cache \
    --do_train --do_eval \
    --vopt \
    --bias_mode mult_then_renorm \
    --train_epochs 3000 \
    --eval_steps 200 \
    --save_steps 10000000 \
    --save_epochs 100 \
    --train_batch_size 1024 \
    --gpu_batch_size 128 \
    --l1 ${L1} \
    --continuing_subword_prefix @@ \
    --task morpheme_prediction \
    --entropic 10.0 \
    --entropy_start 1500 \
    --entropy_end 2250 \
    --max_blocks 1 \
    --max_block_length 12 \
    --max_unit_length 9 \
    --specials "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" \
    --quiet

done
done
done
done
done
