#!/usr/bin/env bash
# set -e
# set +x
# set +v

PY_VERSION=$(python --version)
PY_REQUIRED_VERSION=3.7

if [[ $PY_VERSION = *$PY_REQUIRED_VERSION* ]]  ; then
    echo "Python $PY_REQUIRED_VERSION is installed ;)"
else
    echo "Error. Python $PY_REQUIRED_VERSION is not installed (version found: $PY_VERSION)."
    return 0
fi


REPO='git@bitbucket.org:semanticweb/okgraph.git'
SEPARATOR='\n\n\n##################################\n'
BASE_PATH=$(eval echo ~$USER)/okgraph_clone_and_experiments
REPO_ROOT_PATH=$BASE_PATH/repo
MAGNITUDE_MODELS_PATH=$BASE_PATH/models

echo '##################################'
echo '##################################'
echo '########      STARTED     ########'
echo '##################################'
echo '##################################'

mkdir -p $BASE_PATH
mkdir -p $MAGNITUDE_MODELS_PATH
mkdir -p $REPO_ROOT_PATH
cd $BASE_PATH

echo -e $SEPARATOR 'Repo path: ' $REPO_ROOT_PATH
echo -e $SEPARATOR 'Cloning...'
if [[ ! -z "$(ls -A $REPO_ROOT_PATH)" ]]; then
    echo -e $SEPARATOR 'A repository already exists on ' $REPO_ROOT_PATH
    read -p "Do you want to stop the scritp? (y/[n]) " -n 1 -r
    echo    # (optional) move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    git clone $REPO --branch feature/se_optimization_based_algo $REPO_ROOT_PATH
fi


echo -e $SEPARATOR 'Creating venv...'
python3.7 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools devtools

echo -e $SEPARATOR 'Installing required libraries...'
echo $REPO_ROOT_PATH
cd $REPO_ROOT_PATH
pip install -r requirements.txt # this may take several minutes

echo -e $SEPARATOR 'Installing okgraph...'
python setup.py install



mkdir -p $REPO_ROOT_PATH/okgraph/task/set_expansion/optimum/experiment/models
MAGNITUDE_MODEL_POSITION=$MAGNITUDE_MODELS_PATH/GoogleNews-vectors-negative300.magnitude
if [[ ! -e $MAGNITUDE_MODEL_POSITION ]]; then
    echo -e $SEPARATOR 'Downloading magnitude file(s)...'
    wget http://magnitude.plasticity.ai/word2vec/medium/GoogleNews-vectors-negative300.magnitude -O $MAGNITUDE_MODEL_POSITION
fi
if [[ ! -e $NEW_MAGNITUDE_FILE ]]; then
    echo -e $SEPARATOR 'Creating magnitude file link...'
    NEW_MAGNITUDE_FILE=$REPO_ROOT_PATH/okgraph/task/set_expansion/optimum/experiment/models/GoogleNews-vectors-negative300.magnitude
    ln -s $MAGNITUDE_MODEL_POSITION $NEW_MAGNITUDE_FILE
fi

echo -e $SEPARATOR 'End.'
