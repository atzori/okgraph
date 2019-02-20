#!/usr/bin/env bash


wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip
mv text8 text8.txt
python tests/corpus_to_model.py -i text8.txt -o text8.txt.bin
python -m pymagnitude.converter -i text8.txt.bin -o text8.txt.magnitude

