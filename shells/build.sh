#!/bin/bash
# This script checks if the directory content has changed since the last build.   
# If it has, it rebuilds the program
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
