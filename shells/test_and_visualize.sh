#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <maze_path> <model_path> <record_path>"
    echo "Example: $0 Proportion_Maze_26_26_3 26_26_3 target.path"
    exit 1
fi

maze_path=$1
model_path=$2
record_path=$3
log_file="logs/visualize.log"
{
    python eval_rl_select.py \
        --map_path "$maze_path" \
        --model_path "$model_path" \
        --selector "all" \
        --probability 1.0 \
        --eval_times 1 \
        --record \
        --record_path "recordings/$record_path.paths"
} | tee -a "$log_file"

python MAPF_visualize/visualization.py \
    --map_file "MAPF_visualize/maps/$maze_path.map" \
    --path_file "recordings/$record_path.path" \
    --path_format="default"

echo "Visualize finished. Check the visualization at logs"