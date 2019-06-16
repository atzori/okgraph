#!/bin/bash env
# set -e
# set +x
# set +v


REPO='git@bitbucket.org:semanticweb/okgraph.git'
SEPARATOR='\n\n\n##################################\n'
BASE_PATH=$(eval echo ~$USER)/okgraph_clone_and_experiments
REPO_ROOT_PATH=$BASE_PATH/repo
MAGNITUDE_MODELS_PATH=$BASE_PATH/models


read -p "Repo is [" $REPO "]. Do you want to change it? (y/[n]) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    return 1
fi


cd $BASE_PATH
source venv/bin/activate

echo -e $SEPARATOR 'Starting experiments...'
cd $REPO_ROOT_PATH/okgraph/task/set_expansion/optimum/experiment/
# cd /Users/lelleem/okgraph/okgraph/task/set_expansion/optimum/experiment/
python experiments.py


echo -e $SEPARATOR 'End.'
