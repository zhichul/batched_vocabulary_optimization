#!/usr/bin/env bash
#$ -wd /home/zlu39/jhu/bopt/scripts/ptb/exp3-1
#$ -V
#$ -N s3e3-1
#$ -j y -o /export/c01/zlu39/jobs/$JOB_NAME-$JOB_ID.out
#$ -M zlu39@jhu.edu
#$ -m e
#$ -l ram_free=35G,mem_free=35G,gpu=1,hostname=c0*|c1*
#$ -q g.q

source /home/gqin2/scripts/acquire-gpus 1
conda env list
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

EXPID="3-1"
mkdir -p ${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/ptb
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/ptb/exp${EXPID}

rm ${ARTIFACT_PREFIX}/44/0.01/768/full_lattice_log.train.json
rm ${ARTIFACT_PREFIX}/44/0.01/768/full_lattice_log.valid.json

for DAT in train valid
do
for CKPT in 0
do
for SEED in 44
do
for SIZE in 768
do
for GL in 0.01
do
CUDA_VISIBLE_DEVICES=1 python3 -O -um bopt.run \
    --seed ${SEED} \
    --train_dataset ${DATA_PREFIX}/ptb.${DAT}.128.txt \
    --eval_dataset ${DATA_PREFIX}/ptb.${DAT}.128.txt \
    --input_vocab ${DATA_PREFIX}/substring8-vocab-threshold=None.txt \
    --output_vocab ${DATA_PREFIX}/substring8-vocab-threshold=None.txt \
    --config ${SCRIPT_PREFIX}/config${SIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${GL}/${SIZE}/ \
    --overwrite_output_dir \
    --do_eval \
    --vopt \
    --bias_mode albo \
    --train_epochs 150 \
    --eval_epochs 1 \
    --eval_steps 100  \
    --save_epochs 10 \
    --save_steps 200  \
    --train_batch_size 126 \
    --gpu_batch_size 2 \
    --continuing_subword_prefix @@ \
    --task language_modeling \
    --entropic -10.0 \
    --entropy_start -1 \
    --entropy_end 0 \
    --entropy_start_dec 1 \
    --entropy_end_dec 3 \
    --max_blocks 6 \
    --max_block_length 32 \
    --max_unit_length 8 \
    --warmup_epochs 1 \
    --weights_learning_rate 0.0 \
    --debug_fixed_point \
    --group_lasso ${GL} \
    --length_normalized_initialization \
    --constant_normalization 20.91 \
    --log_lattice unit marginal log_prob ent lm_marginal \
    --log_lattice_file ${ARTIFACT_PREFIX}/${SEED}/${GL}/${SIZE}/full_lattice_log.${DAT}.json \
    --log_lattice_key ${CKPT}


done
done
done
done
done

for DAT in train valid
do
for CKPT in 400 800 1200 1600 2000 2400 2800 3200 3600 4000 4400 4800 5200 5600 6000 6400 6800 7200
do
for SEED in 44
do
for SIZE in 768
do
for GL in 0.01
do
CUDA_VISIBLE_DEVICES=1 python3 -O -um bopt.run \
    --seed ${SEED} \
    --train_dataset ${DATA_PREFIX}/ptb.${DAT}.128.txt \
    --eval_dataset ${DATA_PREFIX}/ptb.${DAT}.128.txt \
    --input_vocab ${DATA_PREFIX}/substring8-vocab-threshold=None.txt \
    --output_vocab ${DATA_PREFIX}/substring8-vocab-threshold=None.txt \
    --config ${SCRIPT_PREFIX}/config${SIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${GL}/${SIZE}/ \
    --overwrite_output_dir \
    --do_eval \
    --vopt \
    --bias_mode albo \
    --train_epochs 150 \
    --eval_epochs 1 \
    --eval_steps 100  \
    --save_epochs 10 \
    --save_steps 200  \
    --train_batch_size 126 \
    --gpu_batch_size 2 \
    --continuing_subword_prefix @@ \
    --task language_modeling \
    --entropic -10.0 \
    --entropy_start -1 \
    --entropy_end 0 \
    --entropy_start_dec 1 \
    --entropy_end_dec 3 \
    --max_blocks 6 \
    --max_block_length 32 \
    --max_unit_length 8 \
    --warmup_epochs 1 \
    --weights_learning_rate 0.0 \
    --debug_fixed_point \
    --group_lasso ${GL} \
    --length_normalized_initialization \
    --constant_normalization 20.91 \
    --model_name ${ARTIFACT_PREFIX}/${SEED}/${GL}/${SIZE}/checkpoint-${CKPT} \
    --log_lattice unit marginal log_prob ent lm_marginal \
    --log_lattice_file ${ARTIFACT_PREFIX}/${SEED}/${GL}/${SIZE}/full_lattice_log.${DAT}.json \
    --log_lattice_key ${CKPT}


done
done
done
done
done
