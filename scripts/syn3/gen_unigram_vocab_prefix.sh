#!/usr/bin/env bash
DATA_PREFIX=${BLU_CORPORA}/ptb

for SIZE in 42 92 126 142 192 392 592 792 845
do
spm_train --input=/export/a01/corpora/vopt/syn/3/full/train.txt --model_prefix=/export/a01/corpora/vopt/syn/3/full/spmd-unigram-${SIZE} --vocab_size=${SIZE} --character_coverage=1.0 --model_type=unigram --add_dummy_prefix=true
REAL_SIZE=$((${SIZE} + 8))
TRUNC_SIZE=$((${SIZE} - 3))
cat /export/a01/corpora/vopt/syn/3/full/spmd-unigram-${SIZE}.vocab > /export/a01/corpora/vopt/syn/3/full/spmd-unigram-${SIZE}.txt
cat /export/a01/corpora/vopt/syn/3/full/spmd-unigram-${SIZE}.txt | extract_column 0 | tail -n $TRUNC_SIZE > gen_unigram_vocab_tmp
cat /export/a01/corpora/vopt/syn/3/full/spm-unigram-vocab-header.txt gen_unigram_vocab_tmp > /export/a01/corpora/vopt/syn/3/full/spmd-unigram-vocab-${REAL_SIZE}.txt
cat /export/a01/corpora/vopt/syn/3/full/spmd-unigram-${SIZE}.txt | extract_column 0 1 | tail -n $TRUNC_SIZE > gen_unigram_vocab_tmp
cat /export/a01/corpora/vopt/syn/3/full/spm-unigram-weights-header.txt gen_unigram_vocab_tmp > /export/a01/corpora/vopt/syn/3/full/spmd-unigram-weights-${REAL_SIZE}.txt
rm gen_unigram_vocab_tmp


done
