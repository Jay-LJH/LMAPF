#!/bin/bash
# This script checks if the directory content has changed since the last run.   
# If it has, it rebuilds the program and runs it. Otherwise, it runs the program without rebuilding.
DIR_PATH="PIBT"
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
    cmake ..
    make clean
    make -j8
    cd ../..
    CURRENT_HASH=$(find "$DIR_PATH" -type f -exec sha256sum {} + | sort | sha256sum)
    echo "$CURRENT_HASH" > "$HASH_FILE"
else
    echo "Directory content has not changed. No need to rebuild."
fi

python eval_MAPF.py "$@" --model_path maze-26-26-3 --map_path Proportion_Maze_26_26_3