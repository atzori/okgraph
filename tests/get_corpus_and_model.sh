#!/usr/bin/env bash

# text8 corpus from: http://mattmahoney.net/dc/textdata.html

CORPUS_TO_DOWNLOAD='text8'

if [[ ! -f "text7.head.gz" ]]; then
    python -m gensim.downloader --download ${CORPUS_TO_DOWNLOAD} && # text8.gz [33MB]
    gunzip -k -f ~/gensim-data/text8/text8.gz &&
    head -c 10000000 ~/gensim-data/text8/text8 > text7.head &&
    gzip text7.head
fi

# from [simple text] to [word embeddings]
python corpus_to_model.py -i text7.head.gz -o text7.head.bin &&

# from [word embeddings] to [magnitude]
python -m pymagnitude.converter -i text7.head.bin -o text7.head.magnitude

