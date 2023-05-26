function violin_data() {
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

          python3 -m experiments.scripts.best_dev \
            ${OUTPUT_DIR}/log.json dev_accuracy max \
            step \
            test_accuracy \
            train_accuracy \
            --output_json \
            --add_field seed ${SEED} \
            --add_field data ${DATA} \
            --add_field l1 ${L1} > violin/l1=${L1}.runs.es.tmp

          python3 -m experiments.scripts.json_join \
            --path violin/l1=${L1}.runs.es.tmp \
            --src violin/l1=${L1}.runs.tmp \
            --fields \
            dev_es_accuracy:dev_accuracy \
            train_es_accuracy:train_accuracy \
            test_es_accuracy:test_accuracy \
            --output violin/l1=${L1}.runs.tmp

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
              \
              --output violin/l1=${L1}.runs.tmp
            python3 -m experiments.scripts.json_join \
              --path "${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/checkpoint-early-stopping/${INPUT_NAME}.1best.tokenizations.f1.json" \
              --src violin/l1=${L1}.runs.tmp \
              --fields \
              ${INPUT_NAME}_es_token_precision:token_precision \
              ${INPUT_NAME}_es_token_recall:token_recall \
              ${INPUT_NAME}_es_token_f1:token_f1 \
              ${INPUT_NAME}_es_boundary_precision:boundary_precision \
              ${INPUT_NAME}_es_boundary_recall:boundary_recall \
              ${INPUT_NAME}_es_boundary_f1:boundary_f1 \
              ${INPUT_NAME}_es_unique_predicted_tokens:unique_predicted_tokens \
              ${INPUT_NAME}_es_unique_gold_tokens:unique_gold_tokens \
              \
              ${INPUT_NAME}_es_prefix_token_precision:prefix_token_precision \
              ${INPUT_NAME}_es_prefix_token_recall:prefix_token_recall \
              ${INPUT_NAME}_es_prefix_token_f1:prefix_token_f1 \
              ${INPUT_NAME}_es_prefix_boundary_precision:prefix_boundary_precision \
              ${INPUT_NAME}_es_prefix_boundary_recall:prefix_boundary_recall \
              ${INPUT_NAME}_es_prefix_boundary_f1:prefix_boundary_f1 \
              ${INPUT_NAME}_es_prefix_unique_gold_tokens:prefix_unique_gold_tokens \
              \
              ${INPUT_NAME}_es_stem_token_precision:stem_token_precision \
              ${INPUT_NAME}_es_stem_token_recall:stem_token_recall \
              ${INPUT_NAME}_es_stem_token_f1:stem_token_f1 \
              ${INPUT_NAME}_es_stem_boundary_precision:stem_boundary_precision \
              ${INPUT_NAME}_es_stem_boundary_recall:stem_boundary_recall \
              ${INPUT_NAME}_es_stem_boundary_f1:stem_boundary_f1 \
              ${INPUT_NAME}_es_stem_unique_gold_tokens:stem_unique_gold_tokens \
              \
              ${INPUT_NAME}_es_suffix_token_precision:suffix_token_precision \
              ${INPUT_NAME}_es_suffix_token_recall:suffix_token_recall \
              ${INPUT_NAME}_es_suffix_token_f1:suffix_token_f1 \
              ${INPUT_NAME}_es_suffix_boundary_precision:suffix_boundary_precision \
              ${INPUT_NAME}_es_suffix_boundary_recall:suffix_boundary_recall \
              ${INPUT_NAME}_es_suffix_boundary_f1:suffix_boundary_f1 \
              ${INPUT_NAME}_es_suffix_unique_gold_tokens:suffix_unique_gold_tokens \
              \
              --output violin/l1=${L1}.runs.tmp
            for MODE in 1best lattice
            do
              python3 -m experiments.scripts.json_join \
                --path "${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/checkpoint-final/${INPUT_NAME}.${MODE}.metrics.json" \
                --src violin/l1=${L1}.runs.tmp \
                --fields \
                ${INPUT_NAME}_${MODE}_accuracy:accuracy \
                ${INPUT_NAME}_${MODE}_prefix_accuracy:prefix_acc \
                ${INPUT_NAME}_${MODE}_stem_accuracy:stem_acc \
                ${INPUT_NAME}_${MODE}_suffix_accuracy:suffix_acc \
                --output violin/l1=${L1}.runs.tmp

              python3 -m experiments.scripts.json_join \
                --path "${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/checkpoint-early-stopping/${INPUT_NAME}.${MODE}.metrics.json" \
                --src violin/l1=${L1}.runs.tmp \
                --fields \
                ${INPUT_NAME}_es_${MODE}_accuracy:accuracy \
                ${INPUT_NAME}_es_${MODE}_prefix_accuracy:prefix_acc \
                ${INPUT_NAME}_es_${MODE}_stem_accuracy:stem_acc \
                ${INPUT_NAME}_es_${MODE}_suffix_accuracy:suffix_acc \
                --output violin/l1=${L1}.runs.tmp
            done
            python3 -m experiments.scripts.json_join \
              --path "${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/checkpoint-final/${INPUT_NAME}.1best.results.json" \
              --src violin/l1=${L1}.runs.tmp \
              --fields \
              ${INPUT_NAME}_entropy:entropy \
              --output violin/l1=${L1}.runs.tmp
            python3 -m experiments.scripts.json_join \
              --path "${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/checkpoint-early-stopping/${INPUT_NAME}.1best.results.json" \
              --src violin/l1=${L1}.runs.tmp \
              --fields \
              ${INPUT_NAME}_es_entropy:entropy \
              --output violin/l1=${L1}.runs.tmp
          done
          cat violin/l1=${L1}.runs.tmp >> violin/l1=${L1}.runs.json
        done
      done
    done
  done
  rm -f violin/*.tmp
}

declare -A METRIC_LABELS=(
                           # final ckpt accuracy
                           ["dev_accuracy"]="dev accuracy"
                           ["train_accuracy"]="train accuracy"
                           ["dev_entropy"]="dev entropy"
                           ["train.100_entropy"]="train entropy"
                           ["dev_1best_accuracy"]="dev 1best accuracy"
                           ["train.100_1best_accuracy"]="train 1best accuracy"
                           ["dev_1best_prefix_accuracy"]="dev 1best prefix accuracy"
                           ["train.100_1best_prefix_accuracy"]="train 1best prefix accuracy"
                           ["dev_1best_stem_accuracy"]="dev 1best stem accuracy"
                           ["train.100_1best_stem_accuracy"]="train 1best stem accuracy"
                           ["dev_1best_suffix_accuracy"]="dev 1best suffix accuracy"
                           ["train.100_1best_suffix_accuracy"]="train 1best suffix accuracy"
                           ["dev_lattice_accuracy"]="dev lattice accuracy"
                           ["train.100_lattice_accuracy"]="train lattice accuracy"
                           ["dev_lattice_prefix_accuracy"]="dev lattice prefix accuracy"
                           ["train.100_lattice_prefix_accuracy"]="train lattice prefix accuracy"
                           ["dev_lattice_stem_accuracy"]="dev lattice stem accuracy"
                           ["train.100_lattice_stem_accuracy"]="train lattice stem accuracy"
                           ["dev_lattice_suffix_accuracy"]="dev lattice suffix accuracy"
                           ["train.100_lattice_suffix_accuracy"]="train lattice suffix accuracy"
                           # early-stopping ckpt accuracy
                           ["dev_es_accuracy"]="dev early-stopping accuracy"
                           ["train_es_accuracy"]="train early-stopping accuracy"
                           ["dev_es_entropy"]="dev early-stopping entropy"
                           ["train.100_es_entropy"]="train early-stopping entropy"
                           ["dev_es_1best_accuracy"]="dev early-stopping 1best accuracy"
                           ["train.100_es_1best_accuracy"]="train early-stopping 1best accuracy"
                           ["dev_es_1best_prefix_accuracy"]="dev early-stopping 1best prefix accuracy"
                           ["train.100_es_1best_prefix_accuracy"]="train early-stopping 1best prefix accuracy"
                           ["dev_es_1best_stem_accuracy"]="dev early-stopping 1best stem accuracy"
                           ["train.100_es_1best_stem_accuracy"]="train early-stopping 1best stem accuracy"
                           ["dev_es_1best_suffix_accuracy"]="dev early-stopping 1best suffix accuracy"
                           ["train.100_es_1best_suffix_accuracy"]="train early-stopping 1best suffix accuracy"
                           ["dev_es_lattice_accuracy"]="dev early-stopping lattice accuracy"
                           ["train.100_es_lattice_accuracy"]="train early-stopping lattice accuracy"
                           ["dev_es_lattice_prefix_accuracy"]="dev early-stopping lattice prefix accuracy"
                           ["train.100_es_lattice_prefix_accuracy"]="train early-stopping lattice prefix accuracy"
                           ["dev_es_lattice_stem_accuracy"]="dev early-stopping lattice stem accuracy"
                           ["train.100_es_lattice_stem_accuracy"]="train early-stopping lattice stem accuracy"
                           ["dev_es_lattice_suffix_accuracy"]="dev early-stopping lattice suffix accuracy"
                           ["train.100_es_lattice_suffix_accuracy"]="train early-stopping lattice suffix accuracy"
                           # final checkpoint tokenization
                           ["dev_unique_predicted_tokens"]="dev unique predicted tokens"
                           ["train.100_unique_predicted_tokens"]="train unique predicted tokens"
                           ["dev_unique_gold_tokens"]="dev unique gold tokens"
                           ["train.100_unique_gold_tokens"]="train unique gold tokens"
                           ["dev_prefix_unique_gold_tokens"]="dev prefix unique gold tokens"
                           ["dev_stem_unique_gold_tokens"]="dev stem unique gold tokens"
                           ["dev_suffix_unique_gold_tokens"]="dev suffix unique gold tokens"
                           ["train.100_prefix_unique_gold_tokens"]="train.100 prefix unique gold tokens"
                           ["train.100_stem_unique_gold_tokens"]="train.100 stem unique gold tokens"
                           ["train.100_suffix_unique_gold_tokens"]="train.100 suffix unique gold tokens"

                           ["dev_boundary_f1"]="dev boundary f1"
                           ["dev_boundary_precision"]="dev boundary precision"
                           ["dev_boundary_recall"]="dev boundary recall"
                           ["train.100_boundary_f1"]="train boundary f1"
                           ["train.100_boundary_precision"]="train boundary precision"
                           ["train.100_boundary_recall"]="train boundary recall"
                           ["dev_prefix_boundary_f1"]="dev prefix boundary f1"
                           ["dev_prefix_boundary_precision"]="dev prefix boundary precision"
                           ["dev_prefix_boundary_recall"]="dev prefix boundary recall"
                           ["train.100_prefix_boundary_f1"]="train prefix boundary f1"
                           ["train.100_prefix_boundary_precision"]="train prefix boundary precision"
                           ["train.100_prefix_boundary_recall"]="train prefix boundary recall"
                           ["dev_stem_boundary_f1"]="dev stem boundary f1"
                           ["dev_stem_boundary_precision"]="dev stem boundary precision"
                           ["dev_stem_boundary_recall"]="dev stem boundary recall"
                           ["train.100_stem_boundary_f1"]="train stem boundary f1"
                           ["train.100_stem_boundary_precision"]="train stem boundary precision"
                           ["train.100_stem_boundary_recall"]="train stem boundary recall"
                           ["dev_suffix_boundary_f1"]="dev suffix boundary f1"
                           ["dev_suffix_boundary_precision"]="dev suffix boundary precision"
                           ["dev_suffix_boundary_recall"]="dev suffix boundary recall"
                           ["train.100_suffix_boundary_f1"]="train suffix boundary f1"
                           ["train.100_suffix_boundary_precision"]="train suffix boundary precision"
                           ["train.100_suffix_boundary_recall"]="train suffix boundary recall"
                           # early-stopping tokenization
                           ["dev_es_unique_predicted_tokens"]="dev early-stopping unique predicted tokens"
                           ["train.100_es_unique_predicted_tokens"]="train early-stopping unique predicted tokens"
                           ["dev_es_unique_gold_tokens"]="dev early-stopping unique gold tokens"
                           ["train.100_es_unique_gold_tokens"]="train early-stopping unique gold tokens"
                           ["dev_es_prefix_unique_gold_tokens"]="dev early-stopping prefix unique gold tokens"
                           ["dev_es_stem_unique_gold_tokens"]="dev early-stopping stem unique gold tokens"
                           ["dev_es_suffix_unique_gold_tokens"]="dev early-stopping suffix unique gold tokens"
                           ["train.100_es_prefix_unique_gold_tokens"]="train early-stopping.100 prefix unique gold tokens"
                           ["train.100_es_stem_unique_gold_tokens"]="train early-stopping.100 stem unique gold tokens"
                           ["train.100_es_suffix_unique_gold_tokens"]="train early-stopping.100 suffix unique gold tokens"

                           ["dev_es_boundary_f1"]="dev early-stopping boundary f1"
                           ["dev_es_boundary_precision"]="dev early-stopping boundary precision"
                           ["dev_es_boundary_recall"]="dev early-stopping boundary recall"
                           ["train.100_es_boundary_f1"]="train early-stopping boundary f1"
                           ["train.100_es_boundary_precision"]="train early-stopping boundary precision"
                           ["train.100_es_boundary_recall"]="train early-stopping boundary recall"
                           ["dev_es_prefix_boundary_f1"]="dev early-stopping prefix boundary f1"
                           ["dev_es_prefix_boundary_precision"]="dev early-stopping prefix boundary precision"
                           ["dev_es_prefix_boundary_recall"]="dev early-stopping prefix boundary recall"
                           ["train.100_es_prefix_boundary_f1"]="train early-stopping prefix boundary f1"
                           ["train.100_es_prefix_boundary_precision"]="train early-stopping prefix boundary precision"
                           ["train.100_es_prefix_boundary_recall"]="train early-stopping prefix boundary recall"
                           ["dev_es_stem_boundary_f1"]="dev early-stopping stem boundary f1"
                           ["dev_es_stem_boundary_precision"]="dev early-stopping stem boundary precision"
                           ["dev_es_stem_boundary_recall"]="dev early-stopping stem boundary recall"
                           ["train.100_es_stem_boundary_f1"]="train early-stopping stem boundary f1"
                           ["train.100_es_stem_boundary_precision"]="train early-stopping stem boundary precision"
                           ["train.100_es_stem_boundary_recall"]="train early-stopping stem boundary recall"
                           ["dev_es_suffix_boundary_f1"]="dev early-stopping suffix boundary f1"
                           ["dev_es_suffix_boundary_precision"]="dev early-stopping suffix boundary precision"
                           ["dev_es_suffix_boundary_recall"]="dev early-stopping suffix boundary recall"
                           ["train.100_es_suffix_boundary_f1"]="train early-stopping suffix boundary f1"
                           ["train.100_es_suffix_boundary_precision"]="train early-stopping suffix boundary precision"
                           ["train.100_es_suffix_boundary_recall"]="train early-stopping suffix boundary recall"
                         )
declare -A METRIC_NAMES=(
                           ["dev_accuracy"]="dev_accuracy"
                           ["train_accuracy"]="train_accuracy"
                           ["dev_entropy"]="dev_entropy"
                           ["train.100_entropy"]="train_entropy"
                           ["dev_1best_accuracy"]="dev_1best_accuracy"
                           ["train.100_1best_accuracy"]="train_1best_accuracy"
                           ["dev_1best_prefix_accuracy"]="dev_1best_prefix_accuracy"
                           ["train.100_1best_prefix_accuracy"]="train_1best_prefix_accuracy"
                           ["dev_1best_stem_accuracy"]="dev_1best_stem_accuracy"
                           ["train.100_1best_stem_accuracy"]="train_1best_stem_accuracy"
                           ["dev_1best_suffix_accuracy"]="dev_1best_suffix_accuracy"
                           ["train.100_1best_suffix_accuracy"]="train_1best_suffix_accuracy"
                           ["dev_lattice_accuracy"]="dev_lattice_accuracy"
                           ["train.100_lattice_accuracy"]="train_lattice_accuracy"
                           ["dev_lattice_prefix_accuracy"]="dev_lattice_prefix_accuracy"
                           ["train.100_lattice_prefix_accuracy"]="train_lattice_prefix_accuracy"
                           ["dev_lattice_stem_accuracy"]="dev_lattice_stem_accuracy"
                           ["train.100_lattice_stem_accuracy"]="train_lattice_stem_accuracy"
                           ["dev_lattice_suffix_accuracy"]="dev_lattice_suffix_accuracy"
                           ["train.100_lattice_suffix_accuracy"]="train_lattice_suffix_accuracy"

                           ["dev_es_accuracy"]="dev-es_accuracy"
                           ["train_es_accuracy"]="train-es_accuracy"
                           ["dev_es_entropy"]="dev-es_entropy"
                           ["train.100_es_entropy"]="train-es_entropy"
                           ["dev_es_1best_accuracy"]="dev-es_1best_accuracy"
                           ["train.100_es_1best_accuracy"]="train-es_1best_accuracy"
                           ["dev_es_1best_prefix_accuracy"]="dev-es_1best_prefix_accuracy"
                           ["train.100_es_1best_prefix_accuracy"]="train-es_1best_prefix_accuracy"
                           ["dev_es_1best_stem_accuracy"]="dev-es_1best_stem_accuracy"
                           ["train.100_es_1best_stem_accuracy"]="train-es_1best_stem_accuracy"
                           ["dev_es_1best_suffix_accuracy"]="dev-es_1best_suffix_accuracy"
                           ["train.100_es_1best_suffix_accuracy"]="train-es_1best_suffix_accuracy"
                           ["dev_es_lattice_accuracy"]="dev-es_lattice_accuracy"
                           ["train.100_es_lattice_accuracy"]="train-es_lattice_accuracy"
                           ["dev_es_lattice_prefix_accuracy"]="dev-es_lattice_prefix_accuracy"
                           ["train.100_es_lattice_prefix_accuracy"]="train-es_lattice_prefix_accuracy"
                           ["dev_es_lattice_stem_accuracy"]="dev-es_lattice_stem_accuracy"
                           ["train.100_es_lattice_stem_accuracy"]="train-es_lattice_stem_accuracy"
                           ["dev_es_lattice_suffix_accuracy"]="dev-es_lattice_suffix_accuracy"
                           ["train.100_es_lattice_suffix_accuracy"]="train-es_lattice_suffix_accuracy"

                           ["dev_unique_predicted_tokens"]="dev_unique_predicted_tokens"
                           ["train.100_unique_predicted_tokens"]="train_unique_predicted_tokens"
                           ["dev_unique_gold_tokens"]="dev_unique_gold_tokens"
                           ["train.100_unique_gold_tokens"]="train_unique_gold_tokens"
                           ["dev_prefix_unique_gold_tokens"]="dev_prefix_unique_gold_tokens"
                           ["dev_stem_unique_gold_tokens"]="dev_stem_unique_gold_tokens"
                           ["dev_suffix_unique_gold_tokens"]="dev_suffix_unique_gold_tokens"
                           ["train.100_prefix_unique_gold_tokens"]="train_prefix_unique_gold_tokens"
                           ["train.100_stem_unique_gold_tokens"]="train_stem_unique_gold_tokens"
                           ["train.100_suffix_unique_gold_tokens"]="train_suffix_unique_gold_tokens"

                           ["dev_boundary_f1"]="dev_boundary_f1"
                           ["dev_boundary_precision"]="dev_boundary_precision"
                           ["dev_boundary_recall"]="dev_boundary_recall"
                           ["train.100_boundary_f1"]="train_boundary_f1"
                           ["train.100_boundary_precision"]="train_boundary_precision"
                           ["train.100_boundary_recall"]="train_boundary_recall"
                           ["dev_prefix_boundary_f1"]="dev_prefix_boundary_f1"
                           ["dev_prefix_boundary_precision"]="dev_prefix_boundary_precision"
                           ["dev_prefix_boundary_recall"]="dev_prefix_boundary_recall"
                           ["train.100_prefix_boundary_f1"]="train_prefix_boundary_f1"
                           ["train.100_prefix_boundary_precision"]="train_prefix_boundary_precision"
                           ["train.100_prefix_boundary_recall"]="train_prefix_boundary_recall"
                           ["dev_stem_boundary_f1"]="dev_stem_boundary_f1"
                           ["dev_stem_boundary_precision"]="dev_stem_boundary_precision"
                           ["dev_stem_boundary_recall"]="dev_stem_boundary_recall"
                           ["train.100_stem_boundary_f1"]="train_stem_boundary_f1"
                           ["train.100_stem_boundary_precision"]="train_stem_boundary_precision"
                           ["train.100_stem_boundary_recall"]="train_stem_boundary_recall"
                           ["dev_suffix_boundary_f1"]="dev_suffix_boundary_f1"
                           ["dev_suffix_boundary_precision"]="dev_suffix_boundary_precision"
                           ["dev_suffix_boundary_recall"]="dev_suffix_boundary_recall"
                           ["train.100_suffix_boundary_f1"]="train_suffix_boundary_f1"
                           ["train.100_suffix_boundary_precision"]="train_suffix_boundary_precision"
                           ["train.100_suffix_boundary_recall"]="train_suffix_boundary_recall"

                           ["dev_es_unique_predicted_tokens"]="dev-es_unique_predicted_tokens"
                           ["train.100_es_unique_predicted_tokens"]="train-es_unique_predicted_tokens"
                           ["dev_es_unique_gold_tokens"]="dev-es_unique_gold_tokens"
                           ["train.100_es_unique_gold_tokens"]="train-es_unique_gold_tokens"
                           ["dev_es_prefix_unique_gold_tokens"]="dev-es_prefix_unique_gold_tokens"
                           ["dev_es_stem_unique_gold_tokens"]="dev-es_stem_unique_gold_tokens"
                           ["dev_es_suffix_unique_gold_tokens"]="dev-es_suffix_unique_gold_tokens"
                           ["train.100_es_prefix_unique_gold_tokens"]="train-es_prefix_unique_gold_tokens"
                           ["train.100_es_stem_unique_gold_tokens"]="train-es_stem_unique_gold_tokens"
                           ["train.100_es_suffix_unique_gold_tokens"]="train-es_suffix_unique_gold_tokens"

                           ["dev_es_boundary_f1"]="dev-es_boundary_f1"
                           ["dev_es_boundary_precision"]="dev-es_boundary_precision"
                           ["dev_es_boundary_recall"]="dev-es_boundary_recall"
                           ["train.100_es_boundary_f1"]="train-es_boundary_f1"
                           ["train.100_es_boundary_precision"]="train-es_boundary_precision"
                           ["train.100_es_boundary_recall"]="train-es_boundary_recall"
                           ["dev_es_prefix_boundary_f1"]="dev-es_prefix_boundary_f1"
                           ["dev_es_prefix_boundary_precision"]="dev-es_prefix_boundary_precision"
                           ["dev_es_prefix_boundary_recall"]="dev-es_prefix_boundary_recall"
                           ["train.100_es_prefix_boundary_f1"]="train-es_prefix_boundary_f1"
                           ["train.100_es_prefix_boundary_precision"]="train-es_prefix_boundary_precision"
                           ["train.100_es_prefix_boundary_recall"]="train-es_prefix_boundary_recall"
                           ["dev_es_stem_boundary_f1"]="dev-es_stem_boundary_f1"
                           ["dev_es_stem_boundary_precision"]="dev-es_stem_boundary_precision"
                           ["dev_es_stem_boundary_recall"]="dev-es_stem_boundary_recall"
                           ["train.100_es_stem_boundary_f1"]="train-es_stem_boundary_f1"
                           ["train.100_es_stem_boundary_precision"]="train-es_stem_boundary_precision"
                           ["train.100_es_stem_boundary_recall"]="train-es_stem_boundary_recall"
                           ["dev_es_suffix_boundary_f1"]="dev-es_suffix_boundary_f1"
                           ["dev_es_suffix_boundary_precision"]="dev-es_suffix_boundary_precision"
                           ["dev_es_suffix_boundary_recall"]="dev-es_suffix_boundary_recall"
                           ["train.100_es_suffix_boundary_f1"]="train-es_suffix_boundary_f1"
                           ["train.100_es_suffix_boundary_precision"]="train-es_suffix_boundary_precision"
                           ["train.100_es_suffix_boundary_recall"]="train-es_suffix_boundary_recall"
                         )
declare -A METRIC_YMIN=(
                           ["dev_accuracy"]=0.0
                           ["train_accuracy"]=0.0
                           ["dev_entropy"]=0.0
                           ["train.100_entropy"]=0.0
                           ["dev_1best_accuracy"]=0.0
                           ["train.100_1best_accuracy"]=0.0
                           ["dev_1best_prefix_accuracy"]=0.0
                           ["train.100_1best_prefix_accuracy"]=0.0
                           ["dev_1best_stem_accuracy"]=0.0
                           ["train.100_1best_stem_accuracy"]=0.0
                           ["dev_1best_suffix_accuracy"]=0.0
                           ["train.100_1best_suffix_accuracy"]=0.0
                           ["dev_lattice_accuracy"]=0.0
                           ["train.100_lattice_accuracy"]=0.0
                           ["dev_lattice_prefix_accuracy"]=0.0
                           ["train.100_lattice_prefix_accuracy"]=0.0
                           ["dev_lattice_stem_accuracy"]=0.0
                           ["train.100_lattice_stem_accuracy"]=0.0
                           ["dev_lattice_suffix_accuracy"]=0.0
                           ["train.100_lattice_suffix_accuracy"]=0.0

                           ["dev_es_accuracy"]=0.0
                           ["train_es_accuracy"]=0.0
                           ["dev_es_entropy"]=0.0
                           ["train.100_es_entropy"]=0.0
                           ["dev_es_1best_accuracy"]=0.0
                           ["train.100_es_1best_accuracy"]=0.0
                           ["dev_es_1best_prefix_accuracy"]=0.0
                           ["train.100_es_1best_prefix_accuracy"]=0.0
                           ["dev_es_1best_stem_accuracy"]=0.0
                           ["train.100_es_1best_stem_accuracy"]=0.0
                           ["dev_es_1best_suffix_accuracy"]=0.0
                           ["train.100_es_1best_suffix_accuracy"]=0.0
                           ["dev_es_lattice_accuracy"]=0.0
                           ["train.100_es_lattice_accuracy"]=0.0
                           ["dev_es_lattice_prefix_accuracy"]=0.0
                           ["train.100_es_lattice_prefix_accuracy"]=0.0
                           ["dev_es_lattice_stem_accuracy"]=0.0
                           ["train.100_es_lattice_stem_accuracy"]=0.0
                           ["dev_es_lattice_suffix_accuracy"]=0.0
                           ["train.100_es_lattice_suffix_accuracy"]=0.0

                           ["dev_unique_predicted_tokens"]=0
                           ["train.100_unique_predicted_tokens"]=0
                           ["dev_unique_gold_tokens"]=0
                           ["train.100_unique_gold_tokens"]=0
                           ["dev_prefix_unique_gold_tokens"]=0
                           ["dev_stem_unique_gold_tokens"]=0
                           ["dev_suffix_unique_gold_tokens"]=0
                           ["train.100_prefix_unique_gold_tokens"]=0
                           ["train.100_stem_unique_gold_tokens"]=0
                           ["train.100_suffix_unique_gold_tokens"]=0
                           ["dev_boundary_f1"]=0.0
                           ["dev_boundary_precision"]=0.0
                           ["dev_boundary_recall"]=0.0
                           ["train.100_boundary_f1"]=0.0
                           ["train.100_boundary_precision"]=0.0
                           ["train.100_boundary_recall"]=0.0
                           ["dev_prefix_boundary_f1"]=0.0
                           ["dev_prefix_boundary_precision"]=0.0
                           ["dev_prefix_boundary_recall"]=0.0
                           ["train.100_prefix_boundary_f1"]=0.0
                           ["train.100_prefix_boundary_precision"]=0.0
                           ["train.100_prefix_boundary_recall"]=0.0
                           ["dev_stem_boundary_f1"]=0.0
                           ["dev_stem_boundary_precision"]=0.0
                           ["dev_stem_boundary_recall"]=0.0
                           ["train.100_stem_boundary_f1"]=0.0
                           ["train.100_stem_boundary_precision"]=0.0
                           ["train.100_stem_boundary_recall"]=0.0
                           ["dev_suffix_boundary_f1"]=0.0
                           ["dev_suffix_boundary_precision"]=0.0
                           ["dev_suffix_boundary_recall"]=0.0
                           ["train.100_suffix_boundary_f1"]=0.0
                           ["train.100_suffix_boundary_precision"]=0.0
                           ["train.100_suffix_boundary_recall"]=0.0

                           ["dev_es_unique_predicted_tokens"]=0
                           ["train.100_es_unique_predicted_tokens"]=0
                           ["dev_es_unique_gold_tokens"]=0
                           ["train.100_es_unique_gold_tokens"]=0
                           ["dev_es_prefix_unique_gold_tokens"]=0
                           ["dev_es_stem_unique_gold_tokens"]=0
                           ["dev_es_suffix_unique_gold_tokens"]=0
                           ["train.100_es_prefix_unique_gold_tokens"]=0
                           ["train.100_es_stem_unique_gold_tokens"]=0
                           ["train.100_es_suffix_unique_gold_tokens"]=0
                           ["dev_es_boundary_f1"]=0.0
                           ["dev_es_boundary_precision"]=0.0
                           ["dev_es_boundary_recall"]=0.0
                           ["train.100_es_boundary_f1"]=0.0
                           ["train.100_es_boundary_precision"]=0.0
                           ["train.100_es_boundary_recall"]=0.0
                           ["dev_es_prefix_boundary_f1"]=0.0
                           ["dev_es_prefix_boundary_precision"]=0.0
                           ["dev_es_prefix_boundary_recall"]=0.0
                           ["train.100_es_prefix_boundary_f1"]=0.0
                           ["train.100_es_prefix_boundary_precision"]=0.0
                           ["train.100_es_prefix_boundary_recall"]=0.0
                           ["dev_es_stem_boundary_f1"]=0.0
                           ["dev_es_stem_boundary_precision"]=0.0
                           ["dev_es_stem_boundary_recall"]=0.0
                           ["train.100_es_stem_boundary_f1"]=0.0
                           ["train.100_es_stem_boundary_precision"]=0.0
                           ["train.100_es_stem_boundary_recall"]=0.0
                           ["dev_es_suffix_boundary_f1"]=0.0
                           ["dev_es_suffix_boundary_precision"]=0.0
                           ["dev_es_suffix_boundary_recall"]=0.0
                           ["train.100_es_suffix_boundary_f1"]=0.0
                           ["train.100_es_suffix_boundary_precision"]=0.0
                           ["train.100_es_suffix_boundary_recall"]=0.0
                         )

declare -A METRIC_YMAX=(
                           ["dev_accuracy"]=1.05
                           ["train_accuracy"]=1.05
                           ["dev_entropy"]=1.05
                           ["train.100_entropy"]=1.05
                           ["dev_1best_accuracy"]=1.05
                           ["train.100_1best_accuracy"]=1.05
                           ["dev_1best_prefix_accuracy"]=1.05
                           ["train.100_1best_prefix_accuracy"]=1.05
                           ["dev_1best_stem_accuracy"]=1.05
                           ["train.100_1best_stem_accuracy"]=1.05
                           ["dev_1best_suffix_accuracy"]=1.05
                           ["train.100_1best_suffix_accuracy"]=1.05
                           ["dev_lattice_accuracy"]=1.05
                           ["train.100_lattice_accuracy"]=1.05
                           ["dev_lattice_prefix_accuracy"]=1.05
                           ["train.100_lattice_prefix_accuracy"]=1.05
                           ["dev_lattice_stem_accuracy"]=1.05
                           ["train.100_lattice_stem_accuracy"]=1.05
                           ["dev_lattice_suffix_accuracy"]=1.05
                           ["train.100_lattice_suffix_accuracy"]=1.05

                           ["dev_es_accuracy"]=1.05
                           ["train_es_accuracy"]=1.05
                           ["dev_es_entropy"]=1.05
                           ["train.100_es_entropy"]=1.05
                           ["dev_es_1best_accuracy"]=1.05
                           ["train.100_es_1best_accuracy"]=1.05
                           ["dev_es_1best_prefix_accuracy"]=1.05
                           ["train.100_es_1best_prefix_accuracy"]=1.05
                           ["dev_es_1best_stem_accuracy"]=1.05
                           ["train.100_es_1best_stem_accuracy"]=1.05
                           ["dev_es_1best_suffix_accuracy"]=1.05
                           ["train.100_es_1best_suffix_accuracy"]=1.05
                           ["dev_es_lattice_accuracy"]=1.05
                           ["train.100_es_lattice_accuracy"]=1.05
                           ["dev_es_lattice_prefix_accuracy"]=1.05
                           ["train.100_es_lattice_prefix_accuracy"]=1.05
                           ["dev_es_lattice_stem_accuracy"]=1.05
                           ["train.100_es_lattice_stem_accuracy"]=1.05
                           ["dev_es_lattice_suffix_accuracy"]=1.05
                           ["train.100_es_lattice_suffix_accuracy"]=1.05

                           ["dev_unique_predicted_tokens"]=500
                           ["train.100_unique_predicted_tokens"]=500
                           ["dev_unique_gold_tokens"]=500
                           ["train.100_unique_gold_tokens"]=500
                           ["dev_prefix_unique_gold_tokens"]=500
                           ["dev_stem_unique_gold_tokens"]=500
                           ["dev_suffix_unique_gold_tokens"]=500
                           ["train.100_prefix_unique_gold_tokens"]=500
                           ["train.100_stem_unique_gold_tokens"]=500
                           ["train.100_suffix_unique_gold_tokens"]=500
                           ["dev_boundary_f1"]=1.05
                           ["dev_boundary_precision"]=1.05
                           ["dev_boundary_recall"]=1.05
                           ["train.100_boundary_f1"]=1.05
                           ["train.100_boundary_precision"]=1.05
                           ["train.100_boundary_recall"]=1.05
                           ["dev_prefix_boundary_f1"]=1.05
                           ["dev_prefix_boundary_precision"]=1.05
                           ["dev_prefix_boundary_recall"]=1.05
                           ["train.100_prefix_boundary_f1"]=1.05
                           ["train.100_prefix_boundary_precision"]=1.05
                           ["train.100_prefix_boundary_recall"]=1.05
                           ["dev_stem_boundary_f1"]=1.05
                           ["dev_stem_boundary_precision"]=1.05
                           ["dev_stem_boundary_recall"]=1.05
                           ["train.100_stem_boundary_f1"]=1.05
                           ["train.100_stem_boundary_precision"]=1.05
                           ["train.100_stem_boundary_recall"]=1.05
                           ["dev_suffix_boundary_f1"]=1.05
                           ["dev_suffix_boundary_precision"]=1.05
                           ["dev_suffix_boundary_recall"]=1.05
                           ["train.100_suffix_boundary_f1"]=1.05
                           ["train.100_suffix_boundary_precision"]=1.05
                           ["train.100_suffix_boundary_recall"]=1.05

                           ["dev_es_unique_predicted_tokens"]=500
                           ["train.100_es_unique_predicted_tokens"]=500
                           ["dev_es_unique_gold_tokens"]=500
                           ["train.100_es_unique_gold_tokens"]=500
                           ["dev_es_prefix_unique_gold_tokens"]=500
                           ["dev_es_stem_unique_gold_tokens"]=500
                           ["dev_es_suffix_unique_gold_tokens"]=500
                           ["train.100_es_prefix_unique_gold_tokens"]=500
                           ["train.100_es_stem_unique_gold_tokens"]=500
                           ["train.100_es_suffix_unique_gold_tokens"]=500
                           ["dev_es_boundary_f1"]=1.05
                           ["dev_es_boundary_precision"]=1.05
                           ["dev_es_boundary_recall"]=1.05
                           ["train.100_es_boundary_f1"]=1.05
                           ["train.100_es_boundary_precision"]=1.05
                           ["train.100_es_boundary_recall"]=1.05
                           ["dev_es_prefix_boundary_f1"]=1.05
                           ["dev_es_prefix_boundary_precision"]=1.05
                           ["dev_es_prefix_boundary_recall"]=1.05
                           ["train.100_es_prefix_boundary_f1"]=1.05
                           ["train.100_es_prefix_boundary_precision"]=1.05
                           ["train.100_es_prefix_boundary_recall"]=1.05
                           ["dev_es_stem_boundary_f1"]=1.05
                           ["dev_es_stem_boundary_precision"]=1.05
                           ["dev_es_stem_boundary_recall"]=1.05
                           ["train.100_es_stem_boundary_f1"]=1.05
                           ["train.100_es_stem_boundary_precision"]=1.05
                           ["train.100_es_stem_boundary_recall"]=1.05
                           ["dev_es_suffix_boundary_f1"]=1.05
                           ["dev_es_suffix_boundary_precision"]=1.05
                           ["dev_es_suffix_boundary_recall"]=1.05
                           ["train.100_es_suffix_boundary_f1"]=1.05
                           ["train.100_es_suffix_boundary_precision"]=1.05
                           ["train.100_es_suffix_boundary_recall"]=1.05
                         )

function violin_plot() {
  for METRIC in ${!METRIC_LABELS[@]}
  do
    python3 -m experiments.plotting.violin \
      --output violin/"${METRIC_NAMES[${METRIC}]}".png \
      --violin violin/l1=0.01.runs.json l1=0.01 data ${METRIC} teal \
      --violin violin/l1=0.1.runs.json l1=0.1 data ${METRIC} orange \
      --violin violin/l1=1.0.runs.json l1=1.0 data ${METRIC} red \
      --xmap full 11k \
      --xmap small 2k \
      --width 0.9 \
      --ylim "${METRIC_YMIN[${METRIC}]}" "${METRIC_YMAX[${METRIC}]}"  \
      --xlabel "number of training words" \
      --ylabel "${METRIC_LABELS[${METRIC}]}"
  done
}