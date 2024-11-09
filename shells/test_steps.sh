#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <maze_path> <model_path>"
    echo "Example: $0 Proportion_Maze_26_26_3 26_26_3"
    exit 1
fi

maze_path=$1
model_path=$2

steps=(5120 2000 1000 500 200 100)
log_file="logs/steps.log"

{
    for step in "${steps[@]}"
    do
        python eval_rl_select.py \
            --map_path "$maze_path" \
            --model_path "$model_path" \
            --step "$step"
    done  
} | tee "$log_file"