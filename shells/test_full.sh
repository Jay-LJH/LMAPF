if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <maze_path> <model_path> <probability>"
    echo "Example: $0 Proportion_Maze_26_26_3 26_26_3 0.5"
    exit 1
fi

maze_path=$1
model_path=$2
probability=$3

nohup ./shells/test_steps.sh "$maze_path" "$model_path" &
nohup ./shells/test_select.sh "$maze_path" "$model_path" "$probaility" &
nohup ./shells/test_and_visualize.sh "$maze_path" "$model_path" "tmp" &
wait