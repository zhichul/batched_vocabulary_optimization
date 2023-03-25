#!/usr/bin/env bash
EXPID="r1-1"
mkdir -p ${BLU_ARTIFACTS}/bopt/sentiment140/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/sentiment140
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/sentiment140/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/sentiment140/exp${EXPID}
for SEED in 42
do
for SIZE in 768
do
for L1 in 0.1 # 0.01 1.0
do
for VSIZE in  16000 # 32000
do
CUDA_VISIBLE_DEVICES=0 python3 -O -um bopt.run \
    --seed ${SEED} \
    --train_dataset "${DATA_PREFIX}/training.1600000.processed.noemoticon.utf8.length<150.relabel.random_others.first100k.csv" \
    --eval_dataset "${DATA_PREFIX}/training.1600000.processed.noemoticon.utf8.length<150.relabel.random_500.csv" \
    --test_dataset "${DATA_PREFIX}/testdata.manual.2009.06.14.utf8.length<150.no_neutral.relabel.csv" \
    --input_vocab ${DATA_PREFIX}/spm-unigram-vocab-${VSIZE}.txt \
    --weights_file ${DATA_PREFIX}/spm-unigram-weights-${VSIZE}.txt  \
    --output_vocab ${DATA_PREFIX}/output_vocab.txt \
    --config ${SCRIPT_PREFIX}/config${SIZE}-${VSIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${L1}/${SIZE}/${VSIZE} \
    --overwrite_output_dir \
    --do_train --do_eval \
    --vopt \
    --bias_mode mult_then_renorm \
    --train_epochs 10 \
    --eval_steps 20 \
    --save_steps 10000000 \
    --save_epochs 1 \
    --train_batch_size 1024 \
    --gpu_batch_size 4 \
    --l1 ${L1} \
    --task sentiment_analysis \
    --entropic 10.0 \
    --entropy_start 5 \
    --entropy_end 7 \
    --max_blocks 1 \
    --max_block_length 160 \
    --max_unit_length 8 \
    --specials "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" \
    --quiet \
    --segmentation_dictionary ${SCRIPT_PREFIX}/../../simple/analysis/celex_segmentation.tsv ${SCRIPT_PREFIX}/../../simple/analysis/celex_segmentation_mono.tsv \
    --eval_segmentation \
    --log_attention_statistics \
#    --overwrite_cache \

done
done
done
done
