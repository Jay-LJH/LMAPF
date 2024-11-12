#pragma once
#include "rhcr_class.h"

class RHCR_warehouse : public RHCR_class_pibt_learn
{
public:
    std::unordered_map<int, int> drives_in_induct_stations; // number of robot in each stations
    RHCR_warehouse(int seed, int num_of_robots, int rows, int cols,
                   int env_id, vector<vector<int>> py_map, vector<vector<int>> station_map, std::string project_path);
    ~RHCR_warehouse();
    void initialize_start_locations() override;
    void initialize_goal_locations() override;
    void update_goal_locations() override;
    void update_start_goal(int rl_simulation_window) override;
    void finish_task(int agent_id, int location, int timestep) override;
    void assign_goal(int agent,int pos) override;
    void initialize() override;
    int assign_induct_station(int curr) const;
    int assign_eject_station() const;
    SortingGrid *G;
};
