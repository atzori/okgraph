#!/usr/bin/env bash

# text8 corpus from: http://mattmahoney.net/dc/textdata.html

CORPUS_TO_DOWNLOAD='text8'

if [[ ! -e tests/corpus_to_model.py ]]; then
    echo "You need to be on the root folder of the project to execute this (this script require to execute tests/corpus_to_model.py)."
    return
fi

mkdir -p tests/data/

if [[ ! -e tests/data/text7.head.gz ]]; then
    echo "Downlading corpus..."
    python -m gensim.downloader --download ${CORPUS_TO_DOWNLOAD} && # text8.gz [33MB]
    gunzip -k -f ~/gensim-data/text8/text8.gz &&
    head -c 10000000 ~/gensim-data/text8/text8 > text7.head &&
    gzip text7.head
    mv text7.head.gz tests/data/text7.head.gz
else
    echo "[Corpus] already present."
fi


# from [simple text] to [word embeddings]
if [[ ! -e tests/data/text7.head.bin ]]; then
    echo "[text7.head.bin] not present"
    python tests/corpus_to_model.py -i tests/data/text7.head.gz -o tests/data/text7.head.bin
else
    echo "[Bin file] already present."
fi


# from [word embeddings] to [magnitude]
if [[ ! -e tests/data/text7.head.magnitude ]]; then
    echo "[text7.head.magnitude] not present"
    python -m pymagnitude.converter -i tests/data/text7.head.bin -o tests/data/text7.head.magnitude
else
    echo "[Magnitude] file already present."
fi
