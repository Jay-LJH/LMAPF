#pragma once
#include "rhcr_class.h"
#include "MazeGraph.h"

class RHCR_maze : public RHCR_class_pibt_learn
{
public:
    RHCR_maze(int seed,int num_of_robots, int rows,int cols,int env_id,vector<vector<int>> py_map,vector<vector<int>> station_map,std::string project_path);
    ~RHCR_maze();
    void initialize_start_locations() override;
    void initialize_goal_locations() override;
    void update_goal_locations() override;
    void update_start_goal(int rl_simulation_window) override;
    void finish_task(int agent_id, int location, int timestep) override;
    void assign_goal(int pos) override;
    void initialize() override;
    MazeGraph G;
};