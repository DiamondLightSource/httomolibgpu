#!/bin/bash

PKG_NAME=httomolibgpu
USER=httomo-team
OS=noarch
CONDA_TOKEN=$(cat $HOME/.secrets/my_secret.json)

mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld


$CONDA/bin/conda install conda-build
$CONDA/bin/conda install -c conda-forge anaconda-client

$CONDA/bin/conda build . -c conda-forge -c httomo -c rapidsai --no-test

# upload packages to conda
find $CONDA_BLD_PATH/$OS -name *.tar.bz2 | while read file
do
    echo $file
    $CONDA/bin/anaconda -v --show-traceback --token $CONDA_TOKEN upload --label $LABEL $file --force
done
