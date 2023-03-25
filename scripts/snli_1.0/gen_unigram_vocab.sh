#!/usr/bin/env bash
DATA_PREFIX=${BLU_CORPORA}/snli_1.0

for SIZE in  19989 #15989
do
spm_train --input='/export/a01/corpora/snli_1.0/snli_1.0_train.length<150.ws_split.txt' --model_prefix=/export/a01/corpora/snli_1.0/spm-unigram-${SIZE} --vocab_size=${SIZE} --character_coverage=0.9995 --model_type=unigram
REAL_SIZE=$((${SIZE} + 11))
TRUNC_SIZE=$((${SIZE} - 3))
cat /export/a01/corpora/snli_1.0/spm-unigram-${SIZE}.vocab > /export/a01/corpora/snli_1.0/spm-unigram-${SIZE}.txt #| remove_underscore_inject_double_underscore
cat /export/a01/corpora/snli_1.0/spm-unigram-${SIZE}.txt | extract_column 0 | tail -n $TRUNC_SIZE > gen_unigram_vocab_tmp
cat /export/a01/corpora/snli_1.0/spm-unigram-vocab-header.txt gen_unigram_vocab_tmp > /export/a01/corpora/snli_1.0/spm-unigram-vocab-${REAL_SIZE}.txt
cat /export/a01/corpora/snli_1.0/spm-unigram-${SIZE}.txt | extract_column 0 1 | tail -n $TRUNC_SIZE > gen_unigram_vocab_tmp
cat /export/a01/corpora/snli_1.0/spm-unigram-weights-header.txt gen_unigram_vocab_tmp > /export/a01/corpora/snli_1.0/spm-unigram-weights-${REAL_SIZE}.txt
rm gen_unigram_vocab_tmp


done
