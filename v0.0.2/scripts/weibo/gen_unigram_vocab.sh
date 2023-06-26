#!/usr/bin/env bash
#for MAXLEN in 10
#do
#for MIN_COUNT in 10
#do
#
#substring_extractor --max_length ${MAXLEN} --min_count ${MIN_COUNT} < /export/a01/corpora/weibo/train.txt > gen_substring_vocab_tmp
#cat /export/a01/corpora/weibo/spm-unigram-vocab-header.txt gen_substring_vocab_tmp > /export/a01/corpora/weibo/substring-vocab-max_length=${MAXLEN}-min_count=${MIN_COUNT}.txt
#rm gen_substring_vocab_tmp
#
#done
#done

for MAXLEN in 5
do
for MIN_COUNT in 10
do

substring_extractor --max_length ${MAXLEN} --min_count ${MIN_COUNT} < /export/a01/corpora/weibo/train.txt > gen_substring_vocab_tmp
cat /export/a01/corpora/weibo/spm-unigram-vocab-header.txt gen_substring_vocab_tmp > /export/a01/corpora/weibo/substring-vocab-max_length=${MAXLEN}-min_count=${MIN_COUNT}.txt
rm gen_substring_vocab_tmp

done
done