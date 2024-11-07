## Visualization tool for MAPF

#### Setup Guide

Install the gym_maze:

```
pip install -e .
```

Run the sample code:

```
python visualization.py --map-file <map_file> --path-file <--path-file>
```

#### Modify the IO

Modify \_\_draw_maze (maze_view_2d.py: line 164) to change the color of the map.

Modify load_maze (maze_view_2d.py: line 310) to load different map.
