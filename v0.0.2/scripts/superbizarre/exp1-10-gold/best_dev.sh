#!/usr/bin/env bash
EXPID="1-10"
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}

rm -rf tmp
touch tmp
for SEED in 42 44 46
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/
cat ${OUTPUT_DIR}/log.json >> tmp
done
python3 -m experiments.scripts.best_dev tmp dev_accuracy max test_accuracy
rm -rf tmp