#!/bin/bash

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc

mkdir build 
cd build 

cmake ..
make

# ./image_processing