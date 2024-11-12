#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <maze_path> <model_path> <probility>"
    echo "Example: $0 Proportion_Maze_26_26_3 26_26_3 0.5"
    exit 1
fi

maze_path=$1
model_path=$2
probability=$3

log_file="logs/select_$probability.log"

{
    selectors=("pibt" "BC" "random")  
    for selector in "${selectors[@]}"
    do
        python eval_rl_select.py \
            --map_path "$maze_path" \
            --model_path "$model_path" \
            --selector "$selector" \
            --probability "$probability"
    done  
    python eval_rl_select.py \
        --map_path "$maze_path" \
        --model_path "$model_path" \
        --selector "all" \
        --probability 1.0 
} | tee "$log_file"