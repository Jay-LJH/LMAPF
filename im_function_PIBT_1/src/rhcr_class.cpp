#include "rhcr_class.h"

// init RHCR and create graph, preprocessing and set
// call at python gym/global_reset
RHCR_class_pibt_learn::RHCR_class_pibt_learn(int seed, int num_of_robots, int rows, int cols,
                                             int env_id, vector<vector<int>> py_map, vector<vector<int>> station_map, 
                                             std::string project_path) : num_of_robots(num_of_robots), rows(rows), 
                                             cols(cols), G(), solver(G, seed, rows * cols), seed(seed)
{
    srand(seed);
    G.load_map(py_map, station_map, rows, cols);
    G.preprocessing(project_path, env_id); // calculated heuristic of all goals
    initialize();                          // random choose start location and according to cost chose goal
}

RHCR_class_pibt_learn::~RHCR_class_pibt_learn() {}

// call from python gym/global_reset
// update the start & goal locations of agents
void RHCR_class_pibt_learn::update_start_goal(int rl_simulation_window)
{
    simulation_window = rl_simulation_window;
    update_start_locations(); // use paths defined in move to update start
    update_goal_locations();  // generate new goal+ tansfer into py version
    solver.update(goal_locations);
}

// call from python gym/joint_move
// run pibt solver base on action guide and update paths
// return collision time of all agents during this run
vector<int> RHCR_class_pibt_learn::run_pibt(const vector<vector<int>> &action_guide)
{
    bool succ = solver.run(action_guide); // solve one step
    update_paths(solver.solution);
    return solver.coll_times;
}

// call from python gym/local_reset
// update the system based on the true path in gym env
bool RHCR_class_pibt_learn::update_system(vector<vector<pair<int, int>>> input_path) // update goal
{
    vector<vector<int>> c_input_path(num_of_robots);
    // transfer input path from python to c++
    for (int k = 0; k < num_of_robots; k++)
    {
        for (auto poss : input_path[k])
        {
            c_input_path[k].push_back(poss.first * cols + poss.second);
        }
    }

    // update paths
    for (int k = 0; k < num_of_robots; k++) // update paths using input paths
    {
        int length = min(INT_MAX, (int)c_input_path[k].size());
        paths[k].resize(timestep + length); // LMAPF so start from current timestep
        for (int t = 0; t < length; t++)
        {
            State input_state(c_input_path[k][t], timestep + t, -1); // inmapf_solver here is t not timestep + t
            paths[k][timestep + t] = input_state;
        }
    }

    auto new_finished_tasks = move();

    // update tasks
    for (auto task : new_finished_tasks)
    {
        int id, loc, t;
        std::tie(id, loc, t) = task;
        finished_tasks[id].emplace_back(loc, t);
        num_of_tasks++;
        if (G.types[loc] == "Induct")
        {
            drives_in_induct_stations[loc]--; // the drive will leave the current induct station
        }
    }
    timestep += simulation_window;
    return true;
}

// convert heuristics map to python version, from int to position
std::unordered_map<Position, vector<double>> RHCR_class_pibt_learn::obtaion_heuri_map()
{
    std::unordered_map<Position, vector<double>> rl_h_map;
    for (auto it = G.heuristics.begin(); it != G.heuristics.end(); ++it)
    {
        rl_h_map[int2Pos(it->first)] = it->second;
    }
    return rl_h_map;
}

// initilized start and goal position
void RHCR_class_pibt_learn::initialize()
{
    starts.resize(num_of_robots);
    goal_locations.resize(num_of_robots);
    paths.resize(num_of_robots);
    finished_tasks.resize(num_of_robots);
    rl_agent_poss.resize(num_of_robots);
    rl_agent_goals.resize(num_of_robots);

    for (const auto induct : G.inducts)
    {
        drives_in_induct_stations[induct.second] = 0;
    }

    timestep = 0;
    initialize_start_locations(); // randomly chose empty cell
    initialize_goal_locations();  // assign goal from eject(random) and induct station(according to cost)

    // initialize induct station counter
    for (int k = 0; k < num_of_robots; k++)
    {
        // goals
        int goal = goal_locations[k].back().first; // location of the goal
        if (G.types[goal] == "Induct")
        {
            drives_in_induct_stations[goal]++; // increase task on it
        }
        else if (G.types[goal] != "Eject") // at the beginning we only can only choose eject as goal
        {
            std::cout << "ERROR in the type of goal locations" << std::endl;
            std::cout << "The fiducial type of the goal of agent " << k << " is " << G.types[goal] << std::endl;
            exit(-1);
        }
    }
    tie_breaker = solver.initialize(starts);
}

// random choose empty position to place robot
void RHCR_class_pibt_learn::initialize_start_locations()
{
    int N = G.size();
    std::vector<bool> used(N, false);
    for (int k = 0; k < num_of_robots;)
    {
        int loc = rand() % N;
        if (G.types[loc] != "Obstacle" && !used[loc])
        {
            int orientation = -1;
            starts[k] = State(loc, 0, orientation);
            rl_agent_poss[k] = int2Pos(loc);
            paths[k].emplace_back(starts[k]);
            used[loc] = true;
            finished_tasks[k].emplace_back(loc, 0);
            k++;
        }
    }
}

// set goal location for each agent
void RHCR_class_pibt_learn::initialize_goal_locations()
{
    // Choose random goal locations
    // a close induct location can be a goal location, or
    // any eject locations can be goal locations
    // Goal locations are not necessarily unique
    for (int k = 0; k < num_of_robots; k++)
    {
        int goal;
        if (k % 2 == 0) // to induction
        {
            goal = assign_induct_station(starts[k].location);
            drives_in_induct_stations[goal]++; // tasks on it ++
        }
        else // to ejection
        {
            goal = assign_eject_station();
        }
        goal_locations[k].emplace_back(goal, 0); // location,timestep
        rl_agent_goals[k].emplace_back(int2Pos(goal));
    }
}

// assign induct station to agent based on cost
// cost = distance + c * number of tasks on the vertex
// choose the induct station with the lowest cost
int RHCR_class_pibt_learn::assign_induct_station(int curr) const
{
    int assigned_loc;
    auto min_cost = DBL_MAX;
    for (auto induct : drives_in_induct_stations)
    {
        double cost = G.heuristics.at(induct.first)[curr] + c * induct.second;
        if (cost < min_cost)
        {
            min_cost = cost;
            assigned_loc = induct.first;
        }
    }
    return assigned_loc;
}

// assign eject station to agent, eject station and location is randomly chosen
int RHCR_class_pibt_learn::assign_eject_station() const
{
    int n = rand() % G.ejects.size();
    boost::unordered_map<int, std::list<int>>::const_iterator it = G.ejects.begin();
    std::advance(it, n);                // 访问G.ejects容器中第n个元素（从0开始计数）。然而，由于boost::unordered_map的无序性，"第n个元素"这一说法并没有明确的顺序含义，它仅仅表示按照容器内部顺序的第n个元素。
    int p = rand() % it->second.size(); // advance move the iterator n position
    auto it2 = it->second.begin();
    std::advance(it2, p); // choice one id(location) from the eject station
    return *it2;
}

// update the path of each agent
void RHCR_class_pibt_learn::update_paths(const std::vector<State> &MAPF_paths)
{
    rl_path.clear();
    rl_path.resize(num_of_robots);
    for (int k = 0; k < num_of_robots; k++)
    {
        rl_path[k] = int2Pos(MAPF_paths[k].location);
    }
}

// add states from pibt to paths, and remove goals if the agent has reached them
// return : list(agent_id, location, timestep)
list<tuple<int, int, int>> RHCR_class_pibt_learn::move()
{
    int start_timestep = timestep;
    int end_timestep = timestep + simulation_window;

    list<tuple<int, int, int>> curr_finished_tasks; // <agent_id, location, timestep>

    // Agents waits at its current locations if no future paths are assigned
    // add piddings to paths
    {
        int t = end_timestep;
        for (int k = 0; k < num_of_robots; k++)
        {
            while ((int)paths[k].size() <= t)
            {
                State final_state = paths[k].back();
                paths[k].emplace_back(final_state.location, final_state.timestep + 1, final_state.orientation);
            }
        }
    }

    // remove goals if the agent has reached them and add to finished_tasks 
    for (int t = start_timestep; t <= end_timestep; t++)
    {
        for (int k = 0; k < num_of_robots; k++)
        {
            State curr = paths[k][t];       
            // the agent finish its current task, what is the meaning of first,second
            if (!goal_locations[k].empty() &&
                curr.location == goal_locations[k].front().first &&
                curr.timestep >= goal_locations[k].front().second) 
            {
                goal_locations[k].erase(goal_locations[k].begin());
                curr_finished_tasks.emplace_back(k, curr.location, t);
            }
        }
    }
    return curr_finished_tasks;
}

// add new tasks to agents if needed for the next planning horizon
void RHCR_class_pibt_learn::update_goal_locations()
{
    rl_agent_goals.clear();
    rl_agent_goals.resize(num_of_robots);

    // process each agent
    for (int k = 0; k < num_of_robots; k++)
    {
        pair<int, int> curr(paths[k][timestep].location, timestep); // current location

        pair<int, int> goal;           // The last goal location & id
        if (goal_locations[k].empty()) // all tasks have been completed, the agent should wait
        {
            goal = curr;
        }
        else
        {
            goal = goal_locations[k].back();
        }
        int min_timesteps = G.get_Manhattan_distance(curr.first, goal.first); // cannot use h values, because graph edges may have weights
        min_timesteps = max(min_timesteps, goal.second);                      // what does second mean->the earlest arrive time
        // when the agent might finish its tasks during the next planning horizon 
        // assign new tasks to the agent that keep the agent busy
        while (min_timesteps <= simulation_window)     
        {
            // assign a new task, switch between induct and eject
            int next;
            if (G.types[goal.first] == "Induct") // warehouse task, need to switch between induct and eject
            {
                next = assign_eject_station();
            }
            else if (G.types[goal.first] == "Eject")
            {
                next = assign_induct_station(curr.first);
                drives_in_induct_stations[next]++; // number of tasks on it +++, used to calculate cost during assign new induct goal
            }
            else
            {
                std::cout << "ERROR in update_goal_function()" << std::endl;
                std::cout << "The fiducial type should not be " << G.types[curr.first] << std::endl;
                exit(-1);
            }
            goal_locations[k].emplace_back(next, 0);
            min_timesteps += G.get_Manhattan_distance(next, goal.first); // G.heuristics.at(next)[goal];
            min_timesteps = max(min_timesteps, goal.second);
            goal = make_pair(next, 0);
        }
        for (auto pos : goal_locations[k])
            rl_agent_goals[k].emplace_back(int2Pos(pos.first));
    }
}

// check if more than half of the agents always choose stop action, if so stop the LMAPF task
// check the path of each agent, when it wait during the simulation window, it is a wait agent
bool RHCR_class_pibt_learn::congested() const
{
    if (simulation_window <= 1)
        return false;
    int wait_agents = 0;
    for (const auto &path : paths)
    {
        int t = 0;
        while (t < simulation_window && path[timestep].location == path[timestep + t].location &&
               path[timestep].orientation == path[timestep + t].orientation) // choose wait action
            t++;
        if (t == simulation_window) 
            wait_agents++;
    }
    return wait_agents > num_of_robots / 2;
}

// update the start locations of agents
void RHCR_class_pibt_learn::update_start_locations() // reset timestep to 0
{
    for (int k = 0; k < num_of_robots; k++)
    {
        starts[k] = State(paths[k][timestep].location, 0, paths[k][timestep].orientation);
    }
}
