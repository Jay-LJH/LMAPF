#pragma once
#include <random>
#include "States.h"
#include "BasicGraph.h"
#include "MazeGraph.h"
#include "common.h"
#include "util.h"

class PIBT
{
public:
    vector<int> solution;
    PIBT(vector<vector<int>> &py_map,int num_of_agents,int seed);
    ~PIBT();
    MazeGraph G;
    int num_of_agents;
    vector<int> coll_times;
    int curr_ag_id=0;
    void initialize();
    void update_goal(const vector<int> &goal_locations);
    // main
    bool run(const vector<vector<int>>& action_guide);

private:
  // PIBT agent
  struct Agent {
    int id;
    int v_now;        // current location
    int v_next;       // next location
    int g;            // goal
    int elapsed;        // eta
    int init_d;       // initial distance
    float tie_breaker;  // epsilon, tie-breaker
  };
  vector<int> goal_id;
  using Agents = std::vector<Agent*>;
  Agents A;
  // <node-id, agent>, whether the node is occupied or not
  // work as reservation table
  Agents occupied_now;
  Agents occupied_next;
  vector<vector<int>> action_guide;
  vector< vector<pair<int, int> > > goal_locations;
  std::mt19937* MT;
  // result of priority inheritance: true -> valid, false -> invalid
  bool funcPIBT(Agent* ai, Agent* aj = nullptr);
  bool funcPIBT_without_guide(Agent* ai, Agent* aj = nullptr);

};

