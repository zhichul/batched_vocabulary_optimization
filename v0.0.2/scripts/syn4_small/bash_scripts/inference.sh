function inference_1best() {
  # this function for each run and each checkpoint runs inference in 1best mode and saves
  # predictions and labels. It uses the lattice position embedding scheme.
  # Also saves a .results.json file with entropy and accuracy.
  ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
  SCRIPT_PREFIX=${HOME}/jhu/bopt/v0.0.2/scripts/syn4_small/exp${EXPID}

  for SEED in 42 44 46 48 50 52 54 56 58 60
  do
  for SIZE in 768
  do
  for L1 in 0.01 0.1 1.0
  do
  for DATA in  100 500 small full
  do
  for NAME in train.100 dev test
  do
  for CKPT in checkpoint-early-stopping checkpoint-final
  do
  DATA_PREFIX=${BLU_CORPORA}/vopt/syn/4/${DATA}
  CKPT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/${CKPT}

  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING=1 python3 -O -um bopt.infer \
      --output_directory ${CKPT_DIR} \
      --overwrite_output_directory \
      \
      --model_path ${CKPT_DIR} \
      \
      --task classification \
      --domain morpheme_prediction \
      --dataset ${DATA_PREFIX}/${NAME}.csv \
      --data_num_workers 1 \
      --name ${NAME}.1best \
      \
      --bias_mode ${BIAS_MODE} \
      \
      --input_vocab ${CKPT_DIR}/learned_vocab.txt \
      --input_tokenizer_weights ${CKPT_DIR}/learned_vocab.txt \
      --output_vocab ${DATA_PREFIX}/output_vocab.txt \
      --input_tokenizer_model unigram \
      --input_tokenizer_mode 1best \
      --special_tokens "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" \
      --pad_token "[PAD]" \
      --log_space_parametrization \
      --use_lattice_position_ids \
      \
      --max_blocks 1 \
      --max_unit_length 9 \
      --max_block_length 12 \
      --space_character " " \
      --remove_space \
      --split_on_space \
      --gpu_batch_size 128 \
      --device "cuda"
  done
  done
  done
  done
  done
  done
}

function inference_lattice() {
  # this function for each run and each checkpoint runs inference in lattice mode and saves
  # predictions and labels. Also saves a .results.json file with entropy and accuracy.
  ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
  SCRIPT_PREFIX=${HOME}/jhu/bopt/v0.0.2/scripts/syn4_small/exp${EXPID}

  for SEED in 42 44 46 48 50 52 54 56 58 60
  do
  for SIZE in 768
  do
  for L1 in 0.01 0.1 1.0
  do
  for DATA in  100 500 small full
  do
  for NAME in train.100 dev test
  do
  for CKPT in checkpoint-early-stopping checkpoint-final
  do
  DATA_PREFIX=${BLU_CORPORA}/vopt/syn/4/${DATA}
  CKPT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/${CKPT}

  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING=1 python3 -O -um bopt.infer \
      --output_directory ${CKPT_DIR} \
      --overwrite_output_directory \
      \
      --model_path ${CKPT_DIR} \
      \
      --task classification \
      --domain morpheme_prediction \
      --dataset ${DATA_PREFIX}/${NAME}.csv \
      --data_num_workers 1 \
      --name ${NAME}.lattice \
      \
      --bias_mode ${BIAS_MODE} \
      \
      --input_vocab ${CKPT_DIR}/learned_vocab.txt \
      --input_tokenizer_weights ${CKPT_DIR}/learned_vocab.txt \
      --output_vocab ${DATA_PREFIX}/output_vocab.txt \
      --input_tokenizer_model unigram \
      --input_tokenizer_mode lattice \
      --special_tokens "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" \
      --pad_token "[PAD]" \
      --log_space_parametrization \
      --use_lattice_position_ids \
      \
      --max_blocks 1 \
      --max_unit_length 9 \
      --max_block_length 12 \
      --space_character " " \
      --remove_space \
      --split_on_space \
      --gpu_batch_size 128 \
      --device "cuda"
  done
  done
  done
  done
  done
  done
}

function compute_metrics () {
  # this function for each run and checkpoint computes the task metrics (accuracy, and breakdown accuracies)
  # and saves them to a file
  ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
  SCRIPT_PREFIX=${HOME}/jhu/bopt/v0.0.2/scripts/syn4_small/exp${EXPID}

  rm -rf compute_metrics.tmp
  for SEED in 42 44 46 48 50 52 54 56 58 60
  do
  for SIZE in 768
  do
  for L1 in 0.01 0.1 1.0
  do
  for DATA in  100 500 small full
  do
  for NAME in train.100 dev test
  do
  for CKPT in checkpoint-early-stopping checkpoint-final
  do
  for MODE in 1best lattice
  do
  DATA_PREFIX=${BLU_CORPORA}/vopt/syn/4/${DATA}
  CKPT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/${CKPT}
  echo ${CKPT_DIR}
  python3 -um experiments.scripts.accuracy \
      ${CKPT_DIR}/${NAME}.${MODE}.labels.tsv \
      ${CKPT_DIR}/${NAME}.${MODE}.predictions.tsv \
      --categories_file ${DATA_PREFIX}/${NAME}_categories.tsv \
      --output_file compute_metrics.tmp
  python3 -um experiments.scripts.json_join \
      --path ${CKPT_DIR}/${NAME}.${MODE}.results.json \
      --src compute_metrics.tmp \
      --output ${CKPT_DIR}/${NAME}.${MODE}.metrics.json \
      --fields entropy
  done
  done
  done
  done
  done
  done
  done
  rm -rf compute_metrics.tmp
}