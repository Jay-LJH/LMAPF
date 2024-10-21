#include "rhcr_maze.h"

RHCR_maze::RHCR_maze(int seed, int num_of_robots, int rows, int cols,
                               int env_id, vector<vector<int>> py_map, vector<vector<int>> station_map, std::string project_path) :
                                RHCR_class_pibt_learn(seed, num_of_robots, rows, cols, env_id, py_map, station_map, project_path)
{
    
}