function lattice_hyperopt () {
  # this function optimizes the hyperparameters other than training DATA,
  # and the step (always takes the final iteration, i.e. after entropy regularization)
  # then merges the tokenization metrics in from the final iteration, and saves
  # to hyperopt-results.json
  mkdir -p ${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
  ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}

  rm -f hyperopt-results.json
  touch hyperopt-results.json
  for DATA in 100 500 small full
  do
    # extract results for a specific training data size into a single file
    touch hyperopt.${DATA}.tmp
    for SIZE in 768
    do
      for L1 in 0.01 0.1 1.0
      do
        for SEED in 42 44 46 48 50 52 54 56 58 60
        do
          OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}
          python3 -m experiments.scripts.best_dev \
            ${OUTPUT_DIR}/log.json step max \
            dev_accuracy \
            test_accuracy \
            train_accuracy \
            dev_entroy \
            test_entropy \
            --output_json \
            --add_field seed ${SEED} \
            --add_field l1 ${L1} >> hyperopt.${DATA}.tmp
        done
      done
    done

    # aggregate runs for a specific training data size by maxing over hyperparams
    python3 -m experiments.scripts.best_dev \
      hyperopt.${DATA}.tmp dev_accuracy max \
      dev_accuracy \
      test_accuracy \
      train_accuracy \
      dev_entroy \
      test_entropy \
      l1 \
      seed \
      --output_json \
      --add_field data ${DATA} > hyperopt.${DATA}.results.tmp

    # merge with tokenization results
    for INPUT_NAME in train.100 dev test
    do
      python3 -m experiments.scripts.json_join \
        --path "${ARTIFACT_PREFIX}/{0}/${SIZE}/{1}/${DATA}/checkpoint-final/${INPUT_NAME}.1best.tokenizations.f1.json" \
        --src hyperopt.${DATA}.results.tmp \
        --join_by seed l1 \
        --fields \
        ${INPUT_NAME}_token_precision:token_precision \
        ${INPUT_NAME}_token_recall:token_recall \
        ${INPUT_NAME}_token_f1:token_f1 \
        ${INPUT_NAME}_boundary_precision:boundary_precision \
        ${INPUT_NAME}_boundary_recall:boundary_recall \
        ${INPUT_NAME}_boundary_f1:boundary_f1 \
        ${INPUT_NAME}_unique_predicted_tokens:unique_predicted_tokens \
        ${INPUT_NAME}_unique_gold_tokens:unique_gold_tokens \
        \
        ${INPUT_NAME}_prefix_token_precision:prefix_token_precision \
        ${INPUT_NAME}_prefix_token_recall:prefix_token_recall \
        ${INPUT_NAME}_prefix_token_f1:prefix_token_f1 \
        ${INPUT_NAME}_prefix_boundary_precision:prefix_boundary_precision \
        ${INPUT_NAME}_prefix_boundary_recall:prefix_boundary_recall \
        ${INPUT_NAME}_prefix_boundary_f1:prefix_boundary_f1 \
        ${INPUT_NAME}_prefix_unique_gold_tokens:prefix_unique_gold_tokens \
        \
        ${INPUT_NAME}_stem_token_precision:stem_token_precision \
        ${INPUT_NAME}_stem_token_recall:stem_token_recall \
        ${INPUT_NAME}_stem_token_f1:stem_token_f1 \
        ${INPUT_NAME}_stem_boundary_precision:stem_boundary_precision \
        ${INPUT_NAME}_stem_boundary_recall:stem_boundary_recall \
        ${INPUT_NAME}_stem_boundary_f1:stem_boundary_f1 \
        ${INPUT_NAME}_stem_unique_gold_tokens:stem_unique_gold_tokens \
        \
        ${INPUT_NAME}_suffix_token_precision:suffix_token_precision \
        ${INPUT_NAME}_suffix_token_recall:suffix_token_recall \
        ${INPUT_NAME}_suffix_token_f1:suffix_token_f1 \
        ${INPUT_NAME}_suffix_boundary_precision:suffix_boundary_precision \
        ${INPUT_NAME}_suffix_boundary_recall:suffix_boundary_recall \
        ${INPUT_NAME}_suffix_boundary_f1:suffix_boundary_f1 \
        ${INPUT_NAME}_suffix_unique_gold_tokens:suffix_unique_gold_tokens \
        --output hyperopt.${DATA}.results.tmp
    done
    cat hyperopt.${DATA}.results.tmp >> hyperopt-results.json

    rm -f hyperopt.${DATA}.results.tmp
    rm -f hyperopt.${DATA}.tmp

  done
}