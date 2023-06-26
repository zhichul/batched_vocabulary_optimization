#!/usr/bin/env bash
shopt -s extglob # allow star expansion?
source vars.sh
mkdir -p ${BLU_ARTIFACTS2}/boptv2/syn4_small/exp${EXPID}
ARTIFACT_PREFIX=${BLU_ARTIFACTS2}/boptv2/syn4_small/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/v0.0.2/scripts/syn4_small/exp${EXPID}

for SEED in 46
do
for SIZE in 768
do
for L1 in 0.01 1.0
do
for DATA in 100 500 small full
do

OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/
#STEPS="@($(echo $(seq 0 600) | awk '{ gsub(/ /, "|"); print }'))"

for STEP in $(seq 450 524)
do
rm -r ${OUTPUT_DIR}/dynamics/expanded-dynamics-${STEP}
python3 -O -um bopt.learning_dynamics.post_process \
    --output_directory ${OUTPUT_DIR}/dynamics \
    --inputs ${OUTPUT_DIR}/${STEP}-*-dynamics.pt
done
done
done
done
done
