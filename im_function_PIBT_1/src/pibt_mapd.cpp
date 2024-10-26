#include "pibt_mapd.h"

std::unordered_map<int, int> myMap = {
    {0, 0},
    {1, 1}, //
    {23, 2},
    {-1, 3},
    {-23, 4},
}; // 0 stay, 1 right, 2 down, 3 left, 4 up

PIBT_MAPD::PIBT_MAPD(const BasicGraph &G, int seed, int map_size)
    : G(G),
      occupied_now(Agents(map_size, nullptr)),
      occupied_next(Agents(map_size, nullptr))
{
    this->MT = new std::mt19937(seed);
}
PIBT_MAPD::~PIBT_MAPD()
{
    for (auto a : A)
        delete a;
    A.clear();
    delete MT;
}

vector<float> PIBT_MAPD::initialize(const vector<State> &starts)
{
    num_of_agents = starts.size();
    vector<float> tie_breaker(num_of_agents, 0.0);
    for (int i = 0; i < num_of_agents; ++i)
    {

        Agent *a = new Agent{
            i,                         // id
            starts[i],                 // current location
            State(),                   // next location
            -1,                        // goal
            0,                         // elapsed
            getRandomFloat(0, 1, MT)}; // tie-breaker
        A.push_back(a);
        occupied_now[starts[i].location] = a;
        tie_breaker[i] = a->tie_breaker;
    }
    return tie_breaker;
}

void PIBT_MAPD::update(const vector<vector<pair<int, int>>> &goal_locations)
{
    this->goal_locations = goal_locations;
    goal_id.clear();
    goal_id.resize(num_of_agents, 0);
    // assign goal to agents
    for (auto a : A)
        a->g = goal_locations[a->id][0].first;
}

bool PIBT_MAPD::run(const vector<vector<int>> &action_guide)
{
    auto compare = [this](Agent *a, const Agent *b)
    {
        int d_a = G.heuristics.at(a->g)[a->v_now.location];
        int d_b = G.heuristics.at(b->g)[b->v_now.location];
        if (d_a != d_b)
            return d_a < d_b; //  lower distance represent higher priority
        if (a->elapsed != b->elapsed)
            return a->elapsed > b->elapsed;
        // use initial distance
        return a->tie_breaker > b->tie_breaker;
    };
    coll_times.clear();
    coll_times.resize(num_of_agents, 0);
    // clear previous data
    solution.clear();
    solution.resize(num_of_agents);
    this->action_guide = action_guide;
    std::sort(A.begin(), A.end(), compare);
    for (auto a : A)
    {
        if (a->v_next.location == -1) // if the agent has next location, then skip
            funcPIBT(a);              // determine its next location,aj = nullptr by default
    }
    
    // acting
    // update plan
    for (auto a : A)
    {
        // clear
        if (occupied_now[a->v_now.location] == a)
            occupied_now[a->v_now.location] = nullptr;
        occupied_next[a->v_next.location] = nullptr;

        // set next location
        occupied_now[a->v_next.location] = a;
        // update priority
        a->elapsed = (a->v_next.location == a->g) ? 0 : a->elapsed + 1;
        // reset params
        a->v_now = a->v_next;
        a->v_next = State();
        solution[a->id] = a->v_now;
        if (a->v_now.location == a->g)
        {
            goal_id[a->id]++;
            if (goal_id[a->id] < goal_locations[a->id].size())
                a->g = goal_locations[a->id][goal_id[a->id]].first;
        }
    }

    return true;
}


bool PIBT_MAPD::run()
{
    auto compare = [this](Agent* a, const Agent* b) {
        int d_a = G.heuristics.at(a->g)[a->v_now.location];
        int d_b = G.heuristics.at(b->g)[b->v_now.location];
        if (d_a != d_b) return d_a < d_b;   //  lower distance represent higher priority, true a has higher priority
        if (a->elapsed != b->elapsed) return a->elapsed > b->elapsed;
        // use initial distance
        return a->tie_breaker > b->tie_breaker;
    };
    coll_times.clear();
    coll_times.resize(num_of_agents, 0);
    // clear previous data
    solution.clear();
    solution.resize(num_of_agents);

    std::sort(A.begin(), A.end(), compare);
    for (auto a : A) {
        if (a->v_next.location == -1)              // if the agent has next location, then skip
            funcPIBT_without_guide(a);// determine its next location
        }
    // acting
    // update plan
    for (auto a : A)
    {
        // clear
        if (occupied_now[a->v_now.location] == a) occupied_now[a->v_now.location] = nullptr;
        occupied_next[a->v_next.location] = nullptr;

        // set next location
        occupied_now[a->v_next.location] = a;
        // update priority
        a->elapsed = (a->v_next.location == a->g) ? 0 : a->elapsed + 1;
        // reset params
        a->v_now = a->v_next;
        a->v_next = State();
        solution[a->id]=a->v_now;
        if (a->v_now.location == a->g) {
            goal_id[a->id]++;
            if (goal_id[a->id] < goal_locations[a->id].size())
                a->g = goal_locations[a->id][goal_id[a->id]].first;}
    }
    return true;
}

bool PIBT_MAPD::funcPIBT(Agent *ai, Agent *aj)
{
    // compare two nodes
    auto node_compare = [&](const State &v, const State &u)
    {
        // tie breaker
        int v_action = myMap[v.location - ai->v_now.location];
        int u_action = myMap[u.location - ai->v_now.location];
        if (action_guide[curr_ag_id][v_action] != action_guide[curr_ag_id][u_action])
            return action_guide[curr_ag_id][v_action] > action_guide[curr_ag_id][u_action];
        // cout<<"v action "<<v_action<<"v value "<<action_guide[curr_ag_id][v_action]<<"u action "<<u_action<<"u value "<<action_guide[curr_ag_id][u_action]<<endl;
        int d_v = G.heuristics.at(ai->g)[v.location];
        int d_u = G.heuristics.at(ai->g)[u.location];
        if (d_v != d_u)
            return d_v < d_u; // true means v is better than u
        if (occupied_now[v.location] != nullptr && occupied_now[u.location] == nullptr)
            return false;
        if (occupied_now[v.location] == nullptr && occupied_now[u.location] != nullptr)
            return true;
        // randomize
        return false;
    };
    
    curr_ag_id = ai->id;
    vector<State> C = G.get_neighbors_v(ai->v_now);
    // randomize
    std::shuffle(C.begin(), C.end(), *MT);
    // sort
    std::sort(C.begin(), C.end(), node_compare);

    for (auto u : C)
    {
        // avoid conflicts
        if (occupied_next[u.location] != nullptr)
        {
            coll_times[ai->id]++;
            continue;
        } // vertex collision
        if (aj != nullptr && u.location == aj->v_now.location)
        {
            coll_times[ai->id]++;
            continue;
        } // edge collision

        // reserve
        occupied_next[u.location] = ai;
        ai->v_next = u;

        auto ak = occupied_now[u.location];
        if (ak != nullptr && ak->v_next.location == -1)
        {
            if (!funcPIBT(ak, ai))
            {
                coll_times[ai->id]++;
                continue;
            } // replanning
        }
        // success to plan next one step
        return true;
    }

    // failed to secure node
    occupied_next[ai->v_now.location] = ai;
    ai->v_next = ai->v_now;
    return false;
}

bool PIBT_MAPD::funcPIBT_without_guide(Agent* ai, Agent* aj)
{
  // compare two nodes
    auto node_compare = [&](const State& v, const State& u) {
        // tie breaker
        int d_v = G.heuristics.at(ai->g)[v.location];
        int d_u = G.heuristics.at(ai->g)[u.location];
        if (d_v != d_u) return d_v < d_u;  // true means v is better than u
        if (occupied_now[v.location] != nullptr && occupied_now[u.location] == nullptr)
          return false;
        if (occupied_now[v.location] == nullptr && occupied_now[u.location] != nullptr)
          return true;
        // randomize
        return false;
    };
    curr_ag_id=ai->id;
    vector<State> C = G.get_neighbors_v(ai->v_now);
    // randomize
    std::shuffle(C.begin(), C.end(), *MT);
    // sort
    std::sort(C.begin(), C.end(), node_compare);


    for (auto u : C) {
    // avoid conflicts
    if (occupied_next[u.location] != nullptr)
    {
        coll_times[ai->id]++;
        continue;
    } // vertex collision
    if (aj != nullptr && u.location == aj->v_now.location)
    {
        coll_times[ai->id]++;
        continue;
    }  // edge collision

    // reserve
    occupied_next[u.location] = ai;
    ai->v_next = u;

    auto ak = occupied_now[u.location];
    if (ak != nullptr && ak->v_next.location == -1) {
      if (!funcPIBT_without_guide(ak, ai))
      {
        coll_times[ai->id]++;
        continue; } // replanning
    }
    // success to plan next one step
    return true;
    }

    // failed to secure node
    occupied_next[ai->v_now.location] = ai;
    ai->v_next = ai->v_now;
    return false;
}