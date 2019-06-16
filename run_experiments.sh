#!/bin/bash env
# set -e
# set +x
# set +v


SEPARATOR='\n\n\n##################################\n'
BASE_PATH=$(eval echo ~$USER)/okgraph_clone_and_experiments
REPO_ROOT_PATH=$BASE_PATH/repo
MAGNITUDE_MODELS_PATH=$BASE_PATH/models


cd $BASE_PATH
source venv/bin/activate

echo -e $SEPARATOR 'Starting experiments...'
cd $REPO_ROOT_PATH/okgraph/task/set_expansion/optimum/experiment/
# cd /Users/lelleem/okgraph/okgraph/task/set_expansion/optimum/experiment/
python experiments.py


echo -e $SEPARATOR 'End.'
