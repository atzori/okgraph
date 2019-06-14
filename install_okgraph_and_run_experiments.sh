#!/usr/bin/env bash
set -e

FOLDER_NAME='okgraph_clone_and_experiments'
REPO='git@bitbucket.org:semanticweb/okgraph.git'
SEPARATOR='\n\n\n###\n'

cd /Users/lelleem/Desktop/caccaaa
rm -rf $FOLDER_NAME
rm -rf venv

PROJ_ROOT_PATH=$(pwd)/$FOLDER_NAME/okgraph

echo -e $SEPARATOR 'Created a folder container named: ' $FOLDER_NAME
echo -e $SEPARATOR 'Path: ' $PROJ_ROOT_PATH
mkdir $FOLDER_NAME
cd $FOLDER_NAME

echo -e $SEPARATOR 'Cloning...'
git clone $REPO --branch feature/se_optimization_based_algo

echo -e $SEPARATOR 'Creating venv...'
python3.7 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools devtools

echo -e $SEPARATOR 'Installing required libraries...'
cd $PROJ_ROOT_PATH/okgraph
pip install -r requirements.txt # this may take several minutes

echo -e $SEPARATOR 'Installing okgraph...'
python setup.py install

echo -e $SEPARATOR 'Downloading magnitude file(s)...'
cp /Users/lelleem/jupyter/models/GoogleNews-vectors-negative300.magnitude $PROJ_ROOT_PATH/okgraph/models/
# wget http://magnitude.plasticity.ai/word2vec/medium/GoogleNews-vectors-negative300.magnitude

echo -e $SEPARATOR 'Starting experiments...'
cd $PROJ_ROOT_PATH/okgraph/task/set_expansion/optimum/experiment/
python experiments.py