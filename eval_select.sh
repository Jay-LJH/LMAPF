#!/bin/bash
# This script checks if the directory content has changed since the last run.   
# If it has, it rebuilds the program and runs it. Otherwise, it runs the program without rebuilding.
DIR_PATH="im_function_PIBT_1"
HASH_FILE=".dir_hash"

CURRENT_HASH=$(find "$DIR_PATH" -type f -exec sha256sum {} + | sort | sha256sum)

if [ -f "$HASH_FILE" ]; then
    PREVIOUS_HASH=$(cat "$HASH_FILE")
else
    echo "Initial run, creating hash file."
    echo "$CURRENT_HASH" > "$HASH_FILE"
    PREVIOUS_HASH=""
fi

if [ "$CURRENT_HASH" != "$PREVIOUS_HASH" ]; then
    echo "Directory content has changed. Rebuilding and running the program."
    cd "$DIR_PATH/build"
    cmake ..
    make clean
    make -j8
    cd ../..
    CURRENT_HASH=$(find "$DIR_PATH" -type f -exec sha256sum {} + | sort | sha256sum)
    echo "$CURRENT_HASH" > "$HASH_FILE"
else
    echo "Directory content has not changed. No need to rebuild."
fi
python eval_rl_select.py "$@" all 1.0| tee -a eval_select.txt
python eval_rl_select.py "$@" random 0.5| tee -a eval_select.txt
python eval_rl_select.py "$@" pibt 0.5| tee -a eval_select.txt
python eval_rl_select.py "$@" BC 0.5 | tee -a eval_select.txt