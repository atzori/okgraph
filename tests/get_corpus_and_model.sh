#!/usr/bin/env bash

# text8 corpus from: http://mattmahoney.net/dc/textdata.html

CORPUS_TO_DOWNLOAD='text8'

if [[ ! -e tests/text7.head.gz ]]; then
    echo "Downlading corpus..."
    python -m gensim.downloader --download ${CORPUS_TO_DOWNLOAD} && # text8.gz [33MB]
    gunzip -k -f ~/gensim-data/text8/text8.gz &&
    head -c 10000000 ~/gensim-data/text8/text8 > text7.head &&
    gzip text7.head
    mv text7.head.gz tests/text7.head.gz
else
    echo "[Corpus] already present."
fi


# from [simple text] to [word embeddings]
if [[ ! -e tests/text7.head.bin ]]; then
    echo "[text7.head.bin] not present"
    python tests/corpus_to_model.py -i tests/text7.head.gz -o tests/text7.head.bin
else
    echo "[Bin file] already present."
fi


# from [word embeddings] to [magnitude]
if [[ ! -e tests/text7.head.magnitude.gz ]]; then
    echo "[text7.head.magnitude] not present"
    python -m pymagnitude.converter -i tests/text7.head.bin -o tests/text7.head.magnitude
    gzip tests/text7.head.magnitude
    gunzip tests/text7.head.magnitude.gz
else
    echo "[Magnitude] file already present."
fi
