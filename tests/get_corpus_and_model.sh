#!/usr/bin/env bash

# text8 corpus from: http://mattmahoney.net/dc/textdata.html

CORPUS_TO_DOWNLOAD='text8'

if [[ ! -e tests/corpus_to_model.py ]]; then
    echo "You need to be on the root folder of the project to execute this (this script require to execute tests/corpus_to_model.py)."
    return
fi

mkdir -p tests/data/

if [[ ! -e tests/data/text7.gz ]]; then
    echo "Downlading corpus..."
    pushd .
    cd tests/data
    python -m gensim.downloader --download ${CORPUS_TO_DOWNLOAD} && # text8.gz [33MB] or wget http://mattmahoney.net/dc/text8.zip
    mv ~/gensim-data/text8/text8.gz . &&
    gunzip -f -k text8.gz &&
    head -c 10000000 text8 > text7 &&
    gzip < text7 > text7.gz
    #mv text7.head.gz tests/data/text7.head.gz
    popd
else
    echo "[Corpus] already present."
fi


# from [simple text] to [word embeddings]
if [[ ! -e tests/data/text7.bin ]]; then
    echo "[text7.bin] not present"
    python tests/corpus_to_model.py -i tests/data/text7.gz -o tests/data/text7.bin
else
    echo "[Bin file] already present."
fi


# from [word embeddings] to [magnitude]
if [[ ! -e tests/data/text7.magnitude ]]; then
    echo "[text7.magnitude] not present"
    python -m pymagnitude.converter -i tests/data/text7.bin -o tests/data/text7.magnitude
else
    echo "[Magnitude] file already present."
fi

# from [corpus] to [whoosh]
if [[ ! -e tests/data/indexdir/ ]]; then
    echo "[Folder indexdir not present] not present"
    #gunzip -k -f tests/data/text7.gz
    python -c "from okgraph.indexing import Indexing; index_new=Indexing('tests/data/text7'); tmp=index_new.indexing(name_path='tests/data/indexdir')"
    python -c "from okgraph.utils import creation; tmp=creation('tests/data/text7', name='tests/data/dictTotal.npy', save=True)"
else
    echo "[Indexing whoosh] file already present."
fi
