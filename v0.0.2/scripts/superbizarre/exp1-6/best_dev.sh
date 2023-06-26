#!/usr/bin/env bash
EXPID="1-6"
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}

rm -rf tmp
touch tmp
for SEED in 42 44 46
do
for L1 in 0.1 1.0 0.01
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${L1}
cat ${OUTPUT_DIR}/log.json >> tmp
done
done
python3 -m experiments.scripts.best_dev tmp dev_accuracy max test_accuracy
rm -rf tmp
