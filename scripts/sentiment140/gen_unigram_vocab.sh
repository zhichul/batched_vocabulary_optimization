#!/usr/bin/env bash
DATA_PREFIX=${BLU_CORPORA}/sentiment140

for SIZE in 15989 31989
do
spm_train --input='/export/a01/corpora/sentiment140/training.1600000.processed.noemoticon.latin1.length<150.relabel.random_others.ws_split.txt' --model_prefix=/export/a01/corpora/sentiment140/spm-unigram-${SIZE} --vocab_size=${SIZE} --character_coverage=1.0 --model_type=unigram --add_dummy_prefix false
REAL_SIZE=$((${SIZE} + 11))
TRUNC_SIZE=$((${SIZE} - 3))
cat /export/a01/corpora/sentiment140/spm-unigram-${SIZE}.vocab  > /export/a01/corpora/sentiment140/spm-unigram-${SIZE}.txt #| remove_underscore_inject_double_underscore
cat /export/a01/corpora/sentiment140/spm-unigram-${SIZE}.txt | extract_column 0 | tail -n $TRUNC_SIZE > gen_unigram_vocab_tmp
cat /export/a01/corpora/sentiment140/spm-unigram-vocab-header.txt gen_unigram_vocab_tmp > /export/a01/corpora/sentiment140/spm-unigram-vocab-${REAL_SIZE}.txt
cat /export/a01/corpora/sentiment140/spm-unigram-${SIZE}.txt | extract_column 0 1 | tail -n $TRUNC_SIZE > gen_unigram_vocab_tmp
cat /export/a01/corpora/sentiment140/spm-unigram-weights-header.txt gen_unigram_vocab_tmp > /export/a01/corpora/sentiment140/spm-unigram-weights-${REAL_SIZE}.txt
rm gen_unigram_vocab_tmp


done
