#!/usr/bin/env bash


if [[ ! -d venv/bin ]]; then
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python setup.py install
else
    source venv/bin/activate
    echo "Using cached venv."
fi


if [[ -z "$(ls -A tests/data/)" ]]; then
    source tests/get_corpus_and_model.sh
else
    echo "Using cached corpus and model."
fi
