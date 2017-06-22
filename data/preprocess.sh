#!/bin/bash

# source language suffix (example: en, cs, de, fr)
S=$1

# target language suffix (example: en, cs, de, fr)
T=$2

# path to corpus
CORPUS=$3

# maximum sequence length
MAXLEN=$4

echo "normalizing punctuation.."
perl normalize-punctuation.perl -l ${S} < ${CORPUS}.${S} > ${CORPUS}.norm.${S}
perl normalize-punctuation.perl -l ${T} < ${CORPUS}.${T} > ${CORPUS}.norm.${T}

echo "tokenizing.."
perl tokenizer.perl -l ${S} -threads 10 < ${CORPUS}.norm.${S} > ${CORPUS}.tok.${S}
perl tokenizer.perl -l ${T} -threads 10 < ${CORPUS}.norm.${T} > ${CORPUS}.tok.${T}

echo "learning bpe.."
# learn BPE on joint vocabulary
cat ${CORPUS}.tok.${S} ${CORPUS}.tok.${T} | python subword_nmt/learn_bpe.py -s 30000 > ${S}${T}.bpe

echo "applying bpe.."
python subword_nmt/apply_bpe.py -c ${S}${T}.bpe < ${CORPUS}.tok.${S} > ${CORPUS}.bpe.${S}
python subword_nmt/apply_bpe.py -c ${S}${T}.bpe < ${CORPUS}.tok.${T} > ${CORPUS}.bpe.${T}

echo "cleaning: filtering sequences of length over ${MAXLEN}"
perl clean-corpus-n.perl ${CORPUS}.bpe ${S} ${T} ${CORPUS}.clean 1 ${MAXLEN}

echo "shuffling.."
python shuffle.py ${CORPUS}.clean.${S} ${CORPUS}.clean.${T}

mv ${CORPUS}.clean.${S}.shuf ${CORPUS}.shuf.${S}
mv ${CORPUS}.clean.${T}.shuf ${CORPUS}.shuf.${T}

echo "building dictionaries.."
python build_dictionary.py ${CORPUS}.shuf.${S} ${CORPUS}.shuf.${T}

echo "preprocessing complete.."
python data_statistics.py ${CORPUS}.shuf.${S} ${CORPUS}.shuf.${T}
