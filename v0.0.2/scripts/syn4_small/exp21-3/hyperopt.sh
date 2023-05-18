#!/usr/bin/env bash
EXPID="21-3"
mkdir -p ${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}

rm -f hyperopt-results.json
touch hyperopt-results.json
for POS in token-pos char-pos
do
for DATA in 100 500 small full
do
touch hyperopt.${DATA}.tmp
for SIZE in 768
do
for L1 in 0.01 0.1 1.0
do
for SEED in 42 44 46 48 50 52 54 56 68 60
do
for CKPT in checkpoint-early-stopping checkpoint-final
do
OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/${CKPT}/${POS}
python3 -m experiments.scripts.best_dev \
  ${OUTPUT_DIR}/log.json \
  dev_accuracy max \
  test_accuracy \
  --output_json \
  --add_field l1 ${L1} \
  --add_field ckpt ${CKPT} \
  --add_field seed ${SEED} >> hyperopt.${DATA}.tmp
done
done
done
done
python3 -m experiments.scripts.best_dev \
  hyperopt.${DATA}.tmp dev_accuracy max \
  dev_accuracy \
  test_accuracy \
  l1 \
  seed \
  ckpt \
  --output_json \
  --add_field data ${DATA} > hyperopt.${DATA}.results.tmp \

python3 -m experiments.scripts.json_join \
  --path "${ARTIFACT_PREFIX}/{0}/${SIZE}/{1}/${DATA}/{2}/${POS}/checkpoint-final/test.1best.tokenizations.f1.json" \
  --src hyperopt.${DATA}.results.tmp \
  --join_by seed l1 ckpt \
  --fields \
  token_precision \
  token_recall \
  token_f1 \
  boundary_precision \
  boundary_recall \
  boundary_f1 >> hyperopt-results-${POS}.json

rm -f hyperopt.${DATA}.results.tmp
rm -f hyperopt.${DATA}.tmp

done
done

