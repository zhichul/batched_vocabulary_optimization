#!/usr/bin/env bash
EXPID="21-1"
mkdir -p violin
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}

rm -f violin/*.json
for L1 in 0.01 0.1 1.0
do
  touch violin/l1=${L1}.runs.json
  touch violin/l1=${L1}.runs.tmp
  for DATA in 100 500 small full
  do
    for SIZE in 768
    do
      for SEED in 42 44 46 48 50 52 54 56 58 60
      do
        OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}
        python3 -m experiments.scripts.best_dev \
          ${OUTPUT_DIR}/log.json step max \
          dev_accuracy \
          test_accuracy \
          train_accuracy \
          --output_json \
          --add_field seed ${SEED} \
          --add_field data ${DATA} \
          --add_field l1 ${L1} > violin/l1=${L1}.runs.tmp

        # merge with tokenization results
        for INPUT_NAME in train.100 dev test
        do
          python3 -m experiments.scripts.json_join \
            --path "${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/checkpoint-final/${INPUT_NAME}.1best.tokenizations.f1.json" \
            --src violin/l1=${L1}.runs.tmp \
            --fields \
            ${INPUT_NAME}_token_precision:token_precision \
            ${INPUT_NAME}_token_recall:token_recall \
            ${INPUT_NAME}_token_f1:token_f1 \
            ${INPUT_NAME}_boundary_precision:boundary_precision \
            ${INPUT_NAME}_boundary_recall:boundary_recall \
            ${INPUT_NAME}_boundary_f1:boundary_f1 \
            --output violin/l1=${L1}.runs.tmp
        done
        cat violin/l1=${L1}.runs.tmp >> violin/l1=${L1}.runs.json
      done
    done
  done
done
rm -f violin/*.tmp

