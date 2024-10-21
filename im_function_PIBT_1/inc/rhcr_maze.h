#pragma once
#include "rhcr_class.h"
#include "MazeGraph.h"

class RHCR_maze : public RHCR_class_pibt_learn
{
public:
    MazeGraph* G; 
    // for each position, how many agents want to go there
    // goal_num[position index] = goal_num
    vector<int> goal_num;

    RHCR_maze(int seed, int num_of_robots, int env_id, vector<vector<int>> py_map, std::string project_path);
    ~RHCR_maze();
    void initialize() override;
    void initialize_start_locations() override;
    void initialize_goal_locations() override;
    void update_goal_locations() override;
    void update_start_goal(int rl_simulation_window) override;
    void finish_task(int agent_id, int location, int timestep) override;
    void assign_goal(int agnet,int pos) override;
};