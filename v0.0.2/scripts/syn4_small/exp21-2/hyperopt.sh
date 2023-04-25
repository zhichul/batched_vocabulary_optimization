#!/usr/bin/env bash
EXPID="21-2"
mkdir -p ${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}

rm -f hyperopt-results.json
touch hyperopt-results.json
for DATA in 100 500 small full
do
touch hyperopt.${DATA}.tmp
for SIZE in 768
do
for L1 in 0.01 0.1 1.0
do
for SEED in 42 44 46
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}
python3 -m experiments.scripts.best_dev \
  ${OUTPUT_DIR}/log.json step max \
  dev_accuracy \
  test_accuracy \
  --output_json \
  --add_field seed ${SEED} \
  --add_field l1 ${L1} >> hyperopt.${DATA}.tmp
done
done

python3 -m experiments.scripts.best_dev \
  hyperopt.${DATA}.tmp dev_accuracy max \
  test_accuracy \
  l1 \
  seed \
  --output_json \
  --add_field data ${DATA} > hyperopt.${DATA}.results.tmp \

python3 -m experiments.scripts.json_join \
  --path "${ARTIFACT_PREFIX}/{0}/${SIZE}/{1}/${DATA}/checkpoint-final/test.1best.tokenizations.f1.json" \
  --src hyperopt.${DATA}.results.tmp \
  --join_by seed l1 \
  --fields \
  token_precision \
  token_recall \
  token_f1 \
  boundary_precision \
  boundary_recall \
  boundary_f1 >> hyperopt-results.json

rm -f hyperopt.${DATA}.results.tmp
rm -f hyperopt.${DATA}.tmp

done
done
