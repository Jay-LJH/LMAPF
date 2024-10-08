#pragma once
#include "StateTimeAStar.h"
#include <ctime>
#include "SortingGraph.h"
#include "pibt_mapd.h"

// Base class for MAPF solvers
class RHCR_class_pibt_learn
{
public:
    const int num_of_drives;
    int simulation_window;//h
    const int rows;
    const int cols;
    vector<pair<int, int>> rl_agent_poss;
    vector<vector<pair<int, int>>> rl_agent_goals;
    vector<pair<int, int>> rl_path;
    int num_of_tasks=0; // number of finished tasks
    int timestep;
    int seed;
    int c=8;
    vector<float> tie_breaker;

    // used for MAPF instance
    vector<State> starts;
    vector< vector<pair<int, int> > > goal_locations;
    // record movements of drives
    std::vector<Path> paths;
    std::vector<std::list<std::pair<int, int> > > finished_tasks; // location + finish time

    RHCR_class_pibt_learn(int seed,int num_of_drives, int rows,int cols,int env_id,vector<vector<int>> py_map,vector<vector<int>> station_map,std::string project_path);
    ~RHCR_class_pibt_learn();
    void update_start_goal(int rl_simulation_window);
    vector<int> run_pibt();
    bool update_system(vector<vector<pair<int, int>>> input_path);
    std::map<pair<int,int>, vector<double>> obtaion_heuri_map();
    void update_start_locations();
    bool congested() const;
    void update_paths(const std::vector<State>& MAPF_paths);
    list<tuple<int, int, int>> move();
    void initialize();
    // assign tasks
    void initialize_start_locations();
    void initialize_goal_locations();
    void update_goal_locations();

    int assign_induct_station(int curr) const;
    int assign_eject_station() const;


protected:
    SortingGrid G;
    PIBT_MAPD solver;
    boost::unordered_map<int, int> drives_in_induct_stations;

};
