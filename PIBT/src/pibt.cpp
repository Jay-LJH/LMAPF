#include "pibt.h"

std::unordered_map<int, int> myMap = {
    {0, 0},
    {1, 1}, //
    {26, 2},
    {-1, 3},
    {-26, 4},
}; // 0 stay, 1 right, 2 down, 3 left, 4 up

PIBT::PIBT(vector<vector<int>> &py_map, int num_of_agents, int seed) : 
    G(py_map),occupied_now(Agents(G.size(), nullptr)),
    occupied_next(Agents(G.size(), nullptr))
{
    this->MT = new std::mt19937(seed);
    myMap = {
        {0, 0},
        {1, 1}, //
        {this->G.cols, 2},
        {-1, 3},
        {-this->G.cols, 4},
    }; // 0 stay, 1 right, 2 down, 3 left, 4 up1
    this->num_of_agents = num_of_agents;
    initialize();
    solution.resize(num_of_agents,0);
}
PIBT::~PIBT()
{
    for (auto a : A)
        delete a;
    A.clear();
    delete MT;
}

// initialize the agents start location and goal location
void PIBT::initialize()
{
    int N = G.size();
    std::vector<bool> init(N, false);
    std::vector<bool> goal(N, false);
    for (int k = 0; k < num_of_agents; k++)
    {
        int start_loc = G.get_random_travel();
        while (init[start_loc])
        {
            start_loc = G.get_random_travel();
        }
        int goal_loc = G.get_random_travel();
        while (goal[goal_loc])
        {
            goal_loc = G.get_random_travel();
        }
        Agent *a = new Agent{
            k,                                    // id
            start_loc,                            // current location
            -1,                                   // next location
            goal_loc,                             // goal
            0,                                    // elapsed
            G.heuristics.at(goal_loc)[start_loc], // initial distance
            getRandomFloat(0, 1, MT)};            // tie-breaker
        A.push_back(a);
        tie_breaker.push_back(a->tie_breaker);
        agent_goals.push_back({goal_loc/G.cols, goal_loc%G.cols});
        init[start_loc] = true;
        goal[goal_loc] = true;
        occupied_now[start_loc] = a;
    }
}

void PIBT::update_goal(const vector<int> &goal_locations)
{
    // assign goal to agents
    for (auto a : A)
        a->g = goal_locations[a->id];
}

// action_guide: action_guide[agent_id(num of agent)][action] = priority
// return true if all agents reach their goal
bool PIBT::run(const vector<vector<int>> &action_guide)
{
   auto compare = [this](Agent *a, const Agent *b)
    {
        if (a->elapsed != b->elapsed)
            return a->elapsed > b->elapsed;
        // use initial distance
        if (a->init_d != b->init_d)
            return a->init_d > b->init_d;
        return a->tie_breaker > b->tie_breaker;
    };
    coll_times.clear();
    coll_times.resize(num_of_agents, 0);
    // clear previous data
    this->action_guide = action_guide;
    // sort agents by priority, base on elapsed time, initial distance, and tie-breaker
    std::sort(A.begin(), A.end(), compare);
    for (auto a : A)
    {
        if (a->v_next == -1) // if the agent has next location, then skip
            funcPIBT(a);              // determine its next location,aj = nullptr by default
    }

    // acting
    // update plan
    bool done = true;
    for (auto a : A)
    {
        // clear
        if (occupied_now[a->v_now] == a)
            occupied_now[a->v_now] = nullptr;
        occupied_next[a->v_next] = nullptr;

        // set next location
        occupied_now[a->v_next] = a;
        // update priority
        a->elapsed = (a->v_next == a->g) ? 0 : a->elapsed + 1;
        // reset params
        a->v_now = a->v_next;
        a->v_next = -1;
        solution[a->id] = a->v_now;
        done = done && (a->v_now == a->g);
    }

    return done;
}

bool PIBT::funcPIBT(Agent *ai, Agent *aj)
{
    // compare two nodes
    auto node_compare = [&](const int &v, const int &u)
    {
        // tie breaker
        int v_action = myMap[v - ai->v_now];
        int u_action = myMap[u - ai->v_now];
        if (action_guide[curr_ag_id][v_action] != action_guide[curr_ag_id][u_action])
            return action_guide[curr_ag_id][v_action] > action_guide[curr_ag_id][u_action];
        int d_v = G.heuristics.at(ai->g)[v];
        int d_u = G.heuristics.at(ai->g)[u];
        if (d_v != d_u)
            return d_v < d_u; // true means v is better than u
        if (occupied_now[v] != nullptr && occupied_now[u] == nullptr)
            return false;
        if (occupied_now[v] == nullptr && occupied_now[u] != nullptr)
            return true;
        // randomize
        return false;
    };

    curr_ag_id = ai->id;
    vector<State> C_state = G.get_neighbors_v(ai->v_now);
    vector<int> C;
    for (auto s : C_state)
    {
        C.push_back(s.location);
    }
    // randomize
    std::shuffle(C.begin(), C.end(), *MT);
    // sort
    std::sort(C.begin(), C.end(), node_compare);

    for (auto u : C)
    {
        // avoid conflicts
        if (occupied_next[u] != nullptr)
        {
            coll_times[ai->id]++;
            continue;
        } // vertex collision
        if (aj != nullptr && u == aj->v_now)
        {
            coll_times[ai->id]++;
            continue;
        } // edge collision

        // reserve
        occupied_next[u] = ai;
        ai->v_next = u;

        auto ak = occupied_now[u];
        if (ak != nullptr && ak->v_next == -1)
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
    occupied_next[ai->v_now] = ai;
    ai->v_next = ai->v_now;
    return false;
}

vector<pair<int,int>> PIBT::agent_poss(){
    vector<pair<int,int>> res(num_of_agents);
    for(auto a:A){
        res[a->id] = {a->v_now/G.cols,a->v_now%G.cols};
    }
    return res;
}

auto PIBT::get_heuri_map(){
    unordered_map<pair<int,int>, vector<double>> res;
    for (auto i:G.heuristics){
        int x = i.first/G.cols;
        int y = i.first%G.cols;
        res[{x,y}] = i.second;     
    }
    return res;
}
vector<int> PIBT::elapsed(){
    vector<int> res(num_of_agents);
    for(auto a:A){
        res[a->id] = a->elapsed;
    }
    return res;
}