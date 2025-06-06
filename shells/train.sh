#!/bin/bash
# This script checks if the directory content has changed since the last run.   
# If it has, it rebuilds the program and runs it. Otherwise, it runs the program without rebuilding.
DIR_PATH="im_function_PIBT_1"
HASH_FILE="./shells/.dir_hash"

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
    make clean
    make -j8
    cd ../..
    CURRENT_HASH=$(find "$DIR_PATH" -type f -exec sha256sum {} + | sort | sha256sum)
    echo "$CURRENT_HASH" > "$HASH_FILE"
else
    echo "Directory content has not changed. No need to rebuild."
fi

nohup python driver_agent.py "$@" > ./logs/train.log 2>&1 &