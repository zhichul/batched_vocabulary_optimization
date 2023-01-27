#!/usr/bin/env bash

EXPID="122"
mkdir -p ${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/ptb
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/ptb/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/ptb/exp${EXPID}
for SEED in 44 # 42 46
do
for LAYER in 4 1
do
for HEAD in 4 1
do
for SIZE in 384 96
do
for VSIZE in 10000
do
python3 -O -u /home/blu/jhu/bopt/scripts/simple/analysis/best_dev.py ${ARTIFACT_PREFIX}/${SEED}/${LAYER}/${HEAD}/${SIZE}/${VSIZE}/log.json eval_avg_token test_avg_token
python3 -O -u /home/blu/jhu/bopt/scripts/simple/analysis/best_dev.py ${ARTIFACT_PREFIX}/${SEED}/${LAYER}/${HEAD}/${SIZE}/${VSIZE}/log.json train_loss
#python3 ../../simple/analysis/effectiv_vocab_size.py language_modeling ${ARTIFACT_PREFIX}/${SEED}/${LAYER}/${HEAD}/${SIZE}/${VSIZE}/cache/ptb.train.txt

done
done
done
done
done



# With dropuout enabled during evaluation
#/export/a01/artifacts/bopt/ptb/exp10/44/768/10000/log.json: 3920, avg_token=4.966020234440926
#/export/a01/artifacts/bopt/ptb/exp10/44/768/10000/log.json: 9550, train_loss=1.7776905157986809
#/export/a01/artifacts/bopt/ptb/exp10/44/768/8000/log.json: 3920, avg_token=5.013947724962005
#/export/a01/artifacts/bopt/ptb/exp10/44/768/8000/log.json: 9570, train_loss=1.8004593096281354
#/export/a01/artifacts/bopt/ptb/exp10/44/768/6000/log.json: 3920, avg_token=5.0438119135538635
#/export/a01/artifacts/bopt/ptb/exp10/44/768/6000/log.json: 9570, train_loss=1.8657432104411877
#/export/a01/artifacts/bopt/ptb/exp10/44/768/4000/log.json: 3950, avg_token=5.1177216957262095
#/export/a01/artifacts/bopt/ptb/exp10/44/768/4000/log.json: 9550, train_loss=1.9466594457626343
#/export/a01/artifacts/bopt/ptb/exp10/44/768/2000/log.json: 4610, avg_token=5.227876196394601
#/export/a01/artifacts/bopt/ptb/exp10/44/768/2000/log.json: 9570, train_loss=2.0059987264767027

# with dropout turned off during evaluation
#/export/a01/artifacts/bopt/ptb/exp16/44/768/10000/log.json: 3900, avg_token=4.878713482970931
#/export/a01/artifacts/bopt/ptb/exp16/44/768/10000/log.json: 9600, train_loss=1.7869690065709953
#/export/a01/artifacts/bopt/ptb/exp16/44/768/8000/log.json: 3900, avg_token=4.923476935254924
#/export/a01/artifacts/bopt/ptb/exp16/44/768/8000/log.json: 9600, train_loss=1.8122791836404393
#/export/a01/artifacts/bopt/ptb/exp16/44/768/6000/log.json: 3900, avg_token=4.944433688285667
#/export/a01/artifacts/bopt/ptb/exp16/44/768/6000/log.json: 9600, train_loss=1.875485213393839
#/export/a01/artifacts/bopt/ptb/exp16/44/768/4000/log.json: 4300, avg_token=5.00619448704401
#/export/a01/artifacts/bopt/ptb/exp16/44/768/4000/log.json: 9600, train_loss=1.953520383590307
#/export/a01/artifacts/bopt/ptb/exp16/44/768/2000/log.json: 4900, avg_token=5.08320905846355
#/export/a01/artifacts/bopt/ptb/exp16/44/768/2000/log.json: 9600, train_loss=2.0154149053443193
