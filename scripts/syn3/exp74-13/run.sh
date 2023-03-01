EXPID="74-13"
mkdir -p ${BLU_ARTIFACTS}/bopt/syn3/exp${EXPID}
DATA_PREFIX=${BLU_CORPORA}/vopt/syn/3/full
ARTIFACT_PREFIX=${BLU_ARTIFACTS}/bopt/syn3/exp${EXPID}
SCRIPT_PREFIX=${HOME}/jhu/bopt/scripts/syn3/exp${EXPID}
for SEED in 42
do
for SIZE in 768
do
for L1 in 0.1 #0.01 1.0
do
for LR in 0.2 0.06 #0.02 0.006 0.002 0.0006 0.0002 0.00006
do
CUDA_VISIBLE_DEVICES=0 python3 -O -um bopt.run \
    --seed ${SEED} \
    --train_dataset ${DATA_PREFIX}/train.csv \
    --eval_dataset ${DATA_PREFIX}/train.500.csv \
    --test_dataset ${DATA_PREFIX}/dev.csv \
    --input_vocab ${DATA_PREFIX}/substring-vocab-threshold=None.txt \
    --output_vocab ${DATA_PREFIX}/output_vocab.txt \
    --config ${SCRIPT_PREFIX}/config${SIZE}.json \
    --output_dir ${ARTIFACT_PREFIX}/${SEED}/${SIZE}/${L1}/${LR} \
    --overwrite_output_dir --overwrite_cache \
    --do_train --do_eval \
    --vopt \
    --bias_mode albo \
    --train_epochs 300 \
    --eval_steps 20 \
    --save_steps 10000000 \
    --save_epochs 10 \
    --train_batch_size 1024 \
    --gpu_batch_size 512 \
    --l1 ${L1} \
    --continuing_subword_prefix @@ \
    --task morpheme_prediction \
    --entropic 10.0 \
    --entropy_start 150 \
    --entropy_end 225 \
    --max_blocks 1 \
    --max_block_length 12 \
    --max_unit_length 9 \
    --specials "[UNK]" "[CLS]" "[SEP]" "[PAD]" "[MASK]" "[WBD]" "[SP1]" "[SP2]" "[SP3]" "[SP4]" "[SP5]" \
    --weights_learning_rate ${LR} \
    --eval_segmentation \
    --only_save_vocab \
    --segmentation_dictionary ${DATA_PREFIX}/train.500.csv ${DATA_PREFIX}/dev.csv \
    --log_attention_statistics \
    --quiet \

done
done
done
done
