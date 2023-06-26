function lattice_hyperopt () {
  # this function optimizes the hyperparameters other than training DOMAIN,
  # and the step (always takes the final iteration, i.e. after entropy regularization)
  # then merges the tokenization metrics in from the final iteration, and saves
  # to ${RESULTS_FILE}
  mkdir -p ${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}
  ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}

  rm -f ${RESULTS_FILE}
  touch ${RESULTS_FILE}

  # extract results for a specific training data size into a single file
  touch hyperopt.${DOMAIN}.tmp
  for L1 in 0.01 0.1 1.0
  do
    for SEED in 42 44 46
    do
      OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}//${L1}
      python3 -m experiments.scripts.best_dev \
        ${OUTPUT_DIR}/log.json ${METRIC} ${ORDER} \
        step \
        dev_accuracy \
        test_accuracy \
        dev_entroy \
        test_entropy \
        --output_json \
        --add_field seed ${SEED} \
        --add_field l1 ${L1} >> hyperopt.${DOMAIN}.tmp
    done
  done

  # aggregate runs for a specific training data size by maxing over hyperparams
  python3 -m experiments.scripts.best_dev \
    hyperopt.${DOMAIN}.tmp ${METRIC} ${ORDER} \
    step \
    dev_accuracy \
    test_accuracy \
    dev_entroy \
    test_entropy \
    l1 \
    seed \
    --output_json \
    --add_field data ${DOMAIN} > hyperopt.${DOMAIN}.results.tmp

  # merge with tokenization results
  for INPUT_NAME in dev test
  do
    python3 -m experiments.scripts.json_join \
      --path "${ARTIFACT_PREFIX}/{0}/{1}/checkpoint-final/${DOMAIN}_${INPUT_NAME}.1best.tokenizations.f1.json" \
      --src hyperopt.${DOMAIN}.results.tmp \
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
      --output hyperopt.${DOMAIN}.results.tmp
  done
  cat hyperopt.${DOMAIN}.results.tmp >> ${RESULTS_FILE}

  rm -f hyperopt.${DOMAIN}.results.tmp
  rm -f hyperopt.${DOMAIN}.tmp

}

function lattice_full_attention_hyperopt () {
  # this function optimizes the hyperparameters other than training DOMAIN,
  # and the step (always takes the final iteration, i.e. after entropy regularization)
  # then merges the tokenization metrics in from the final iteration, and saves
  # to ${RESULTS_FILE}
  mkdir -p ${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}
  ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}

  rm -f ${RESULTS_FILE}
  touch ${RESULTS_FILE}

  # extract results for a specific training data size into a single file
  touch hyperopt.${DOMAIN}.tmp
  for SEED in 42 44 46
  do
    OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/
    python3 -m experiments.scripts.best_dev \
      ${OUTPUT_DIR}/log.json ${METRIC} ${ORDER} \
      step \
      dev_accuracy \
      test_accuracy \
      dev_entroy \
      test_entropy \
      --output_json \
      --add_field seed ${SEED} >> hyperopt.${DOMAIN}.tmp
  done

  # aggregate runs for a specific training data size by maxing over hyperparams
  python3 -m experiments.scripts.best_dev \
    hyperopt.${DOMAIN}.tmp ${METRIC} ${ORDER} \
    step \
    dev_accuracy \
    test_accuracy \
    dev_entroy \
    test_entropy \
    seed \
    --output_json \
    --add_field data ${DOMAIN} > hyperopt.${DOMAIN}.results.tmp

  cat hyperopt.${DOMAIN}.results.tmp >> ${RESULTS_FILE}

  rm -f hyperopt.${DOMAIN}.results.tmp
  rm -f hyperopt.${DOMAIN}.tmp

}

function bert_hyperopt () {
  # this function optimizes the hyperparameters other than training DOMAIN,
  # and the step (always takes the final iteration, i.e. after entropy regularization)
  # then merges the tokenization metrics in from the final iteration, and saves
  # to hyperopt-results.json
  mkdir -p ${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}
  ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}

  rm -f hyperopt-results.json
  touch hyperopt-results.json
  # extract results for a specific training data size into a single file
  touch hyperopt.${DOMAIN}.tmp
  for SEED in 42 44 46
  do
    OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}
    python3 -m experiments.scripts.best_dev \
      ${OUTPUT_DIR}/log.json dev_accuracy max \
      step \
      test_accuracy \
      dev_entroy \
      test_entropy \
      --output_json \
      --add_field seed ${SEED} >> hyperopt.${DOMAIN}.tmp
  done

  # aggregate runs for a specific training data size by maxing over hyperparams
  python3 -m experiments.scripts.best_dev \
    hyperopt.${DOMAIN}.tmp dev_accuracy max \
    dev_accuracy \
    test_accuracy \
    dev_entroy \
    test_entropy \
    seed \
    --output_json > hyperopt.${DOMAIN}.results.tmp

  # merge with tokenization results
  for INPUT_NAME in dev test
  do
    python3 -m experiments.scripts.json_join \
      --path "${ARTIFACT_PREFIX}/{0}/checkpoint-final/${DOMAIN}_${INPUT_NAME}.1best.tokenizations.f1.json" \
      --src hyperopt.${DOMAIN}.results.tmp \
      --join_by seed \
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
      --output hyperopt.${DOMAIN}.results.tmp
  done
  cat hyperopt.${DOMAIN}.results.tmp >> hyperopt-results.json

  rm -f hyperopt.${DOMAIN}.results.tmp
  rm -f hyperopt.${DOMAIN}.tmp

}

function bert_finetune_hyperopt () {
  # this function optimizes the hyperparameters other than training DOMAIN,
  # and the step (always takes the final iteration, i.e. after entropy regularization)
  # then merges the tokenization metrics in from the final iteration, and saves
  # to hyperopt-results.json
  mkdir -p ${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}
  ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}

  rm -f hyperopt-results.json
  touch hyperopt-results.json
  # extract results for a specific training data size into a single file
  touch hyperopt.${DOMAIN}.tmp
  for CKPT in checkpoint-final
  do
  for L1 in 0.01 0.1 1.0
  do
    for SEED in 42 44 46
    do
      OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${L1}/${CKPT}
      python3 -m experiments.scripts.best_dev \
        ${OUTPUT_DIR}/log.json dev_accuracy max \
        step \
        dev_accuracy \
        test_accuracy \
        dev_entroy \
        test_entropy \
        --output_json \
        --add_field seed ${SEED} \
        --add_field l1 ${L1} \
        --add_field ckpt ${CKPT} >> hyperopt.${DOMAIN}.tmp
    done
  done
  done

  # aggregate runs for a specific training data size by maxing over hyperparams
  python3 -m experiments.scripts.best_dev \
    hyperopt.${DOMAIN}.tmp dev_accuracy max \
    dev_accuracy \
    test_accuracy \
    dev_entroy \
    test_entropy \
    seed \
    ckpt \
    --output_json > hyperopt.${DOMAIN}.results.tmp


  cat hyperopt.${DOMAIN}.results.tmp >> hyperopt-results.json

  rm -f hyperopt.${DOMAIN}.results.tmp
  rm -f hyperopt.${DOMAIN}.tmp

}


function bert_gold_hyperopt () {
  # this function optimizes the hyperparameters other than training DOMAIN,
  # and the step (always takes the final iteration, i.e. after entropy regularization)
  # then merges the tokenization metrics in from the final iteration, and saves
  # to hyperopt-results.json
  mkdir -p ${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}
  ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/superbizarre/exp${EXPID}

  rm -f hyperopt-results.json
  touch hyperopt-results.json
  # extract results for a specific training data size into a single file
  touch hyperopt.${DOMAIN}.tmp
  for SEED in 42 44 46
  do
    OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}
    python3 -m experiments.scripts.best_dev \
      ${OUTPUT_DIR}/log.json dev_accuracy max \
      step \
      test_accuracy \
      dev_entroy \
      test_entropy \
      --output_json \
      --add_field seed ${SEED} >> hyperopt.${DOMAIN}.tmp
  done

  # aggregate runs for a specific training data size by maxing over hyperparams
  python3 -m experiments.scripts.best_dev \
    hyperopt.${DOMAIN}.tmp dev_accuracy max \
    dev_accuracy \
    test_accuracy \
    dev_entroy \
    test_entropy \
    seed \
    --output_json > hyperopt.${DOMAIN}.results.tmp


  cat hyperopt.${DOMAIN}.results.tmp >> hyperopt-results.json

  rm -f hyperopt.${DOMAIN}.results.tmp
  rm -f hyperopt.${DOMAIN}.tmp

}