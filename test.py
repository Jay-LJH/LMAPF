from im_function_PIBT_1.build import lifelong_pibt_1
from map_generator import *
import alg_parameters

maze = Maze()
print(maze)
rhcr = lifelong_pibt_1.RHCR_maze(42, 10, 1, maze.matrix, ".")
rhcr.update_start_goal(EnvParameters.H)