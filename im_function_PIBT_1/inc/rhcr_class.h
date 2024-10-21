#pragma once
#include "StateTimeAStar.h"
#include <ctime>
#include "SortingGraph.h"
#include "pibt_mapd.h"

// Base class for MAPF solvers
class RHCR_class_pibt_learn
{
public:
    string path;
    int env_id;
    const int num_of_robots; // agent number
    int simulation_window;   // replanning period h
    const int rows;
    const int cols;
    vector<Position> rl_agent_poss;
    vector<vector<Position>> rl_agent_goals;
    vector<Position> rl_path;
    int num_of_tasks = 0; // number of finished tasks
    int timestep;
    int seed;
    int c = 8;
    vector<float> tie_breaker;

    // used for MAPF instance
    vector<State> starts;
    // in a life-long environment, agents have different goals at different time. pair<int, int> is <goal, timestep>
    vector<vector<pair<int, int>>> goal_locations;
    // record movements of robots
    std::vector<Path> paths;
    std::vector<std::list<std::pair<int, int>>> finished_tasks; // location + finish time

    RHCR_class_pibt_learn(int seed, int num_of_robots, int rows, int cols, int env_id,
                          std::string project_path);
    virtual ~RHCR_class_pibt_learn();

    vector<int> run_pibt(const vector<vector<int>> &action_guide);
    bool update_system(vector<vector<pair<int, int>>> input_path);
    std::unordered_map<Position, vector<double>> obtaion_heuri_map();
    void update_start_locations();
    bool congested() const;
    void update_paths(const std::vector<State> &MAPF_paths);
    list<tuple<int, int, int>> move();
    virtual void initialize();
    // assign tasks
    virtual void update_start_goal(int rl_simulation_window);
    virtual void initialize_start_locations();
    virtual void initialize_goal_locations();
    virtual void update_goal_locations();
    virtual void assign_goal(int agent,int pos) { return; };
    virtual void finish_task(int agent_id, int location, int timestep) { return; };
    Position int2Pos(int loc)
    {
        return Position(loc / cols, loc % cols);
    }

protected:
    BasicGraph *G; // it should be a pointer to basegraph, but keep it for convenience
    PIBT_MAPD *solver;
};
