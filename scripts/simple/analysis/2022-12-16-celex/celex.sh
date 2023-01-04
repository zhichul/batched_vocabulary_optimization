#!/usr/bin/env bash


for version in ../celex_segmentation.tsv ../celex_segmentation_permissive.tsv
do
  echo ${version}
  echo "simple english"
  echo "train"
  python3 ../match_celex.py ${version} /export/a01/corpora/simple/se.train.txt /export/a01/artifacts/bopt/simple/exp1-20/42/0.01/768/se.train.txt.viterbi.txt
  python3 ../match_celex.py ${version} /export/a01/corpora/simple/se.train.txt /export/a01/artifacts/bopt/simple/exp2-14/44/768/se.train.txt.viterbi.txt
  python3 ../match_celex.py ${version} /export/a01/corpora/simple/se.train.txt /export/a01/corpora/simple/se.train.sentencepiece.txt
  echo "-------------------------------------------------------------------------------"
  echo "valid"
  python3 ../match_celex.py ${version} /export/a01/corpora/simple/se.valid.txt /export/a01/artifacts/bopt/simple/exp1-20/42/0.01/768/se.valid.txt.viterbi.txt
  python3 ../match_celex.py ${version} /export/a01/corpora/simple/se.valid.txt /export/a01/artifacts/bopt/simple/exp2-14/44/768/se.valid.txt.viterbi.txt
    python3 ../match_celex.py ${version} /export/a01/corpora/simple/se.valid.txt /export/a01/corpora/simple/se.valid.sentencepiece.txt
  echo "==============================================================================="
  echo "penn treebank"
  python3 ../match_celex.py ${version} /export/a01/corpora/ptb/ptb.train.txt /export/a01/artifacts/bopt/ptb/exp3-1/44/0.01/768/ptb.train.txt.viterbi.txt
  python3 ../match_celex.py ${version} /export/a01/corpora/ptb/ptb.train.txt /export/a01/artifacts/bopt/ptb/exp4-1/44/768/ptb.train.txt.viterbi.txt
  python3 ../match_celex.py ${version} /export/a01/corpora/ptb/ptb.train.txt /export/a01/corpora/ptb/ptb.train.sentencepiece.txt
  echo "-------------------------------------------------------------------------------"
  echo "valid"
  python3 ../match_celex.py ${version} /export/a01/corpora/ptb/ptb.valid.txt /export/a01/artifacts/bopt/ptb/exp3-1/44/0.01/768/ptb.valid.txt.viterbi.txt
  python3 ../match_celex.py ${version} /export/a01/corpora/ptb/ptb.valid.txt /export/a01/artifacts/bopt/ptb/exp4-1/44/768/ptb.valid.txt.viterbi.txt
  python3 ../match_celex.py ${version} /export/a01/corpora/ptb/ptb.valid.txt /export/a01/corpora/ptb/ptb.valid.sentencepiece.txt



echo "################################################################################"
done

for version in ../celex_segmentation_permissive.tsv
do
  echo ${version}
  echo "simple english"
  python3 ../match_celex.py ${version} /export/a01/corpora/simple/se.train.txt /export/a01/artifacts/bopt/simple/exp1-20/42/0.01/768/se.train.txt.viterbi.txt
  python3 ../match_celex.py ${version} /export/a01/corpora/simple/se.train.txt /export/a01/artifacts/bopt/simple/exp2-14/44/768/se.train.txt.viterbi.txt
  python3 ../match_celex.py ${version} /export/a01/corpora/simple/se.train.txt /export/a01/corpora/simple/se.train.sentencepiece.txt
  echo "-------------------------------------------------------------------------------"
  python3 ../match_celex.py ${version} /export/a01/corpora/simple/se.valid.txt /export/a01/artifacts/bopt/simple/exp1-20/42/0.01/768/se.valid.txt.viterbi.txt yes
  python3 ../match_celex.py ${version} /export/a01/corpora/simple/se.valid.txt /export/a01/artifacts/bopt/simple/exp2-14/44/768/se.valid.txt.viterbi.txt yes
  python3 ../match_celex.py ${version} /export/a01/corpora/simple/se.valid.txt /export/a01/corpora/simple/se.valid.sentencepiece.txt yes

  echo "==============================================================================="
  echo "penn treebank"
  echo "train"
  python3 ../match_celex.py ${version} /export/a01/corpora/ptb/ptb.train.txt /export/a01/artifacts/bopt/ptb/exp3-1/44/0.01/768/ptb.train.txt.viterbi.txt
  python3 ../match_celex.py ${version} /export/a01/corpora/ptb/ptb.train.txt /export/a01/artifacts/bopt/ptb/exp4-1/44/768/ptb.train.txt.viterbi.txt
  python3 ../match_celex.py ${version} /export/a01/corpora/ptb/ptb.train.txt /export/a01/corpora/ptb/ptb.train.sentencepiece.txt
  echo "-------------------------------------------------------------------------------"
  echo "valid"
  python3 ../match_celex.py ${version} /export/a01/corpora/ptb/ptb.valid.txt /export/a01/artifacts/bopt/ptb/exp3-1/44/0.01/768/ptb.valid.txt.viterbi.txt yes yes
  python3 ../match_celex.py ${version} /export/a01/corpora/ptb/ptb.valid.txt /export/a01/artifacts/bopt/ptb/exp4-1/44/768/ptb.valid.txt.viterbi.txt yes
  python3 ../match_celex.py ${version} /export/a01/corpora/ptb/ptb.valid.txt /export/a01/corpora/ptb/ptb.valid.sentencepiece.txt yes

echo "################################################################################"
done

