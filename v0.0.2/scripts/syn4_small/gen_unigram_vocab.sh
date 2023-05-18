#!/usr/bin/env bash
for DATA in 100 500 small full
do
for SIZE in 42 92 126 142 192 392
do

spm_train --input=/export/a01/corpora/vopt/syn/4/${DATA}/train.txt --model_prefix=/export/a01/corpora/vopt/syn/4/${DATA}/spm-unigram-${SIZE} --vocab_size=${SIZE} --character_coverage=1.0 --model_type=unigram --add_dummy_prefix=false
REAL_SIZE=$((${SIZE} + 8))
TRUNC_SIZE=$((${SIZE} - 3))
cat /export/a01/corpora/vopt/syn/4/${DATA}/spm-unigram-${SIZE}.vocab > /export/a01/corpora/vopt/syn/4/${DATA}/spm-unigram-${SIZE}.txt
cat /export/a01/corpora/vopt/syn/4/${DATA}/spm-unigram-${SIZE}.txt | extract_column 0 | tail -n $TRUNC_SIZE > gen_unigram_vocab_tmp
cat /export/a01/corpora/vopt/syn/4/${DATA}/spm-unigram-vocab-header.txt gen_unigram_vocab_tmp > /export/a01/corpora/vopt/syn/4/${DATA}/spm-unigram-vocab-${REAL_SIZE}.txt
cat /export/a01/corpora/vopt/syn/4/${DATA}/spm-unigram-${SIZE}.txt | extract_column 0 1 | tail -n $TRUNC_SIZE > gen_unigram_vocab_tmp
cat /export/a01/corpora/vopt/syn/4/${DATA}/spm-unigram-weights-header.txt gen_unigram_vocab_tmp > /export/a01/corpora/vopt/syn/4/${DATA}/spm-unigram-weights-${REAL_SIZE}.txt
rm gen_unigram_vocab_tmp
done



substring_extractor --max_length 9 < /export/a01/corpora/vopt/syn/4/${DATA}/train.txt > gen_substring_vocab_tmp
cat /export/a01/corpora/vopt/syn/4/${DATA}/spm-unigram-vocab-header.txt gen_substring_vocab_tmp > /export/a01/corpora/vopt/syn/4/${DATA}/substring-vocab-max_length=9-min_count=1.txt
rm gen_substring_vocab_tmp

substring_extractor --max_length 9 --add_prefix_space "▁" < /export/a01/corpora/vopt/syn/4/${DATA}/train.txt > gen_substring_vocab_tmp
cat /export/a01/corpora/vopt/syn/4/${DATA}/spm-unigram-vocab-header.txt gen_substring_vocab_tmp > /export/a01/corpora/vopt/syn/4/${DATA}/substring-vocab-max_length=9-min_count=1-space="▁".txt
rm gen_substring_vocab_tmp
done