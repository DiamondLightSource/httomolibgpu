#!/bin/bash

set -xe 
cp -rv "$RECIPE_DIR/../../httomolib/cuda_kernels/" "$SRC_DIR/httomolib"

cd $SRC_DIR

python -m pip install .

