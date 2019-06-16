#!/bin/bash env
# set -e
# set -o errexit
# set +x
# set +v

PY_VERSION=$(python3.7 --version)
PY_REQUIRED_VERSION=3.7
NEEDED_PACKAGES='build-essential python3.7 python3.7-dev python3-pip python3-setuptools python3.7-venv'

if [[ $PY_VERSION = *$PY_REQUIRED_VERSION* ]]  ; then
    echo "Python $PY_REQUIRED_VERSION is installed ;)"
else
    echo "Error. Python $PY_REQUIRED_VERSION is not installed (version found: $PY_VERSION)."
    echo "On Ubuntu 18 you can run the following command to install or upgrade:"
    echo "sudo apt install $NEEDED_PACKAGES"
    return 1
fi


REPO_SSH='git@bitbucket.org:semanticweb/okgraph.git'
REPO_HTTPS='https://bitbucket.org/semanticweb/okgraph.git'
SEPARATOR='\n\n\n##################################\n'
BASE_PATH=$(eval echo ~$USER)/okgraph_clone_and_experiments
REPO_ROOT_PATH=$BASE_PATH/repo
MAGNITUDE_MODELS_PATH=$BASE_PATH/models

echo '##################################'
echo '##################################'
echo '########      STARTED     ########'
echo '##################################'
echo '##################################'

echo '1) [SSH] ' $REPO_SSH
echo '2) [HTTPS] ' $REPO_HTTPS
read -p "Wich protocol do you prefer?" -n 1 -r
echo
if [[ $REPLY =~ ^[1]$ ]]; then
    REPO=$REPO_SSH
elif [[ $REPLY =~ ^[2]$ ]]; then
    REPO=$REPO_HTTPS
else
    return 1
fi


mkdir -p $BASE_PATH
mkdir -p $MAGNITUDE_MODELS_PATH
mkdir -p $REPO_ROOT_PATH
cd $BASE_PATH

echo -e $SEPARATOR 'Repo path: ' $REPO_ROOT_PATH

if [[ ! -z "$(ls -A $REPO_ROOT_PATH)" ]]; then
    read -p "Do you want a fresh installation (remove repo and venv)? (y/[n]) " -n 1 -r
    echo    # (optional) move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf $REPO_ROOT_PATH
        rm -rf $BASE_PATH/venv
    fi
fi


if [[ ! -z "$(ls -A $REPO_ROOT_PATH)" ]]; then
    echo -e $SEPARATOR 'A repository already exists on ' $REPO_ROOT_PATH
    read -p "Do you want to stop the scritp? (y/[n]) " -n 1 -r
    echo    # (optional) move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 1
    fi
else
    echo -e $SEPARATOR 'Cloning...'
    git clone $REPO --branch feature/se_optimization_based_algo $REPO_ROOT_PATH
    if [ ! $? -eq 0 ]; then
        echo FAILED
        return 1
    fi
fi



echo -e $SEPARATOR 'Creating venv...'
python3.7 -m venv venv
if [ ! $? -eq 0 ]; then
    echo FAILED
    echo "On Ubuntu 18 you can run the following command to install or upgrade:"
    echo "sudo apt install $NEEDED_PACKAGES"
    return 1
fi

source venv/bin/activate
if [ ! $? -eq 0 ]; then
    echo FAILED
    return 1
fi


echo -e $SEPARATOR 'Upgrading tools...'
pip install --upgrade pip setuptools devtools
if [ ! $? -eq 0 ]; then
    echo FAILED
    return 1
fi


echo -e $SEPARATOR 'Installing required libraries...'
echo $REPO_ROOT_PATH
cd $REPO_ROOT_PATH
pip install -r requirements.txt # this may take several minutes
if [ ! $? -eq 0 ]; then
    echo FAILED
    return 1
fi


echo -e $SEPARATOR 'Installing okgraph...'
python setup.py install
if [ ! $? -eq 0 ]; then
    echo FAILED
    return 1
fi



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
