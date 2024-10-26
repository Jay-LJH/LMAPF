#!/bin/bash
DIR_PATH="im_function_PIBT_1"
cd "$DIR_PATH/build"
cmake ..
make clean
make -j8
cd ../..