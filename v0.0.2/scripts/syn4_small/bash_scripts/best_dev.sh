function best_dev () {
  # this function prints the best iteration based on dev accuracy of log.json for each run
  mkdir -p ${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
  ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}

  for SIZE in 768
  do
  for DATA in 100 500 small full
  do
  for L1 in 0.01 0.1 1.0
  do
  for SEED in 42 44 46 48 50 52 54 56 58 60
  do
  OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}
  python3 -m experiments.scripts.best_dev ${OUTPUT_DIR}/log.json dev_accuracy max
  done
  done
  echo ""
  done
  done
}

function best_dev_position_exp () {
  # this function prints the best iteration based on dev accuracy of log.json for each run
  mkdir -p ${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}
  ARTIFACT_PREFIX=${BLU_ARTIFACTS}/boptv2/syn4_small/exp${EXPID}

  for SIZE in 768
  do
  for DATA in 100 500 small full
  do
  for L1 in 0.01 0.1 1.0
  do
  for SEED in 42 44 46 48 50 52 54 56 58 60
  do
  for CKPT_LOAD in checkpoint-early-stopping checkpoint-final
  do
  for POS in token-pos char-pos
  do
  OUTPUT_DIR=${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${DATA}/${CKPT_LOAD}/${POS}
  python3 -m experiments.scripts.best_dev ${OUTPUT_DIR}/log.json dev_accuracy max
  done
  done
  echo ""
  done
  done
}