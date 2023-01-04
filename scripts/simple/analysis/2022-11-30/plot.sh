#!/usr/bin/env bash


N=100
#EXP=1-24
#MODEL=/export/a01/artifacts/bopt/simple/exp${EXP}/42/0.01/768/checkpoint-1100
#python3 ../extract_pos_embedding_similarity.py ${MODEL} ${EXP} ${N}
#
#EXP=2-14-d
#MODEL=/export/a01/artifacts/bopt/simple/exp${EXP}/42/768/checkpoint-500
#python3 ../extract_pos_embedding_similarity.py ${MODEL} ${EXP} ${N}
#
#EXP=2-14
#for CKPT in 100 200 300 400 500 600 700 800 900 1000
#do
#MODEL=/export/a01/artifacts/bopt/simple/exp${EXP}/44/768/checkpoint-${CKPT}
#echo ${MODEL}
#python3 ../extract_pos_embedding_similarity.py ${MODEL} ${EXP}-${CKPT} ${N}
#done
#
#EXP=2-14-l
#for CKPT in 100 200 300 400 500 600 700 800 900 1000
#do
#MODEL=/export/a01/artifacts/bopt/simple/exp${EXP}/42/768/checkpoint-${CKPT}
#echo ${MODEL}
#python3 ../extract_pos_embedding_similarity.py ${MODEL} ${EXP}-${CKPT} ${N}
#done

EXP=2-12
for CKPT in 500 1000 1500 2000 2500 3000 3500 4000 4500
do

EXP=2-12
MODEL=/export/a01/artifacts/bopt/ptb/exp${EXP}/44/768/checkpoint-${CKPT}
python3 ../extract_pos_embedding_similarity.py ${MODEL} ptb-${EXP}-${CKPT} 256

done