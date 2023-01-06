#!/usr/bin/env bash
DATA_PREFIX=${BLU_CORPORA}/ptb

for SIZE in 990 1990 2990 3990 4990 5990 6990 7990 8990 9990
do
spm_train --input=/export/a01/corpora/ptb/ptb.train.txt --model_prefix=/export/a01/corpora/ptb/spm-unigram-${SIZE} --vocab_size=${SIZE} --character_coverage=1.0 --model_type=unigram
REAL_SIZE=$((${SIZE} + 10))
TRUNC_SIZE=$((${SIZE} - 4))
cat /export/a01/corpora/ptb/spm-unigram-${SIZE}.vocab | remove_underscore > /export/a01/corpora/ptb/spm-unigram-${SIZE}.txt
cat /export/a01/corpora/ptb/spm-unigram-${SIZE}.txt | extract_column 0 | tail -n $TRUNC_SIZE > gen_unigram_vocab_tmp
cat /export/a01/corpora/ptb/spm-unigram-vocab-header.txt gen_unigram_vocab_tmp > /export/a01/corpora/ptb/spm-unigram-vocab-${REAL_SIZE}.txt
cat /export/a01/corpora/ptb/spm-unigram-${SIZE}.txt | extract_column 0 1 | tail -n $TRUNC_SIZE > gen_unigram_vocab_tmp
cat /export/a01/corpora/ptb/spm-unigram-weights-header.txt gen_unigram_vocab_tmp > /export/a01/corpora/ptb/spm-unigram-weights-${REAL_SIZE}.txt
rm gen_unigram_vocab_tmp


done
