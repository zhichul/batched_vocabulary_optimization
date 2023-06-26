#!/usr/bin/env bash
source vars.sh
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
rm -rf tmp1
touch tmp1
python3 -m experiments.scripts.best_dev tmp dev_accuracy max \
 test_accuracy \
  --output_json > tmp1

for INPUT_NAME in dev test
do
python3 -m experiments.scripts.json_join \
      --path "${ARTIFACT_PREFIX}/{0}/{1}/${DATA}/checkpoint-early-stopping/${INPUT_NAME}.1best.tokenizations.f1.json" \
      --src hyperopt.${DATA}.results.tmp \
      --join_by seed l1 \
      --fields \
      ${INPUT_NAME}_boundary_precision:boundary_precision \

done
rm -rf tmp
rm -rf tmp1

