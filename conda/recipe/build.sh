#!/bin/bash

set -xe 
cp -rv "$RECIPE_DIR/../../httomolibgpu/cuda_kernels/" "$SRC_DIR/httomolibgpu"

cd $SRC_DIR

python -m pip install .

