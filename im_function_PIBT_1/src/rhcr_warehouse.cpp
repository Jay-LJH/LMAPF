#include "rhcr_warehouse.h"

RHCR_warehouse::RHCR_warehouse(int seed, int num_of_robots, int rows, int cols,
                               int env_id, vector<vector<int>> py_map, vector<vector<int>> station_map, std::string project_path) :
                                RHCR_class_pibt_learn(seed, num_of_robots, rows, cols, env_id,project_path)                               
{
    RHCR_class_pibt_learn::G =new SortingGrid();
    solver = new PIBT_MAPD(*RHCR_class_pibt_learn::G, seed, rows * cols);
    this->G = static_cast<SortingGrid*>(RHCR_class_pibt_learn::G);
    this->env_id = env_id;
    this->path = project_path;
    srand(seed);
    this->G->load_map(py_map, station_map, rows, cols);
    this->G->preprocessing(project_path, env_id); // calculated heuristic of all goals
    for (const auto induct : this->G->inducts)
    {
        drives_in_induct_stations[induct.second] = 0;
    }
    initialize();                          // random choose start location and according to cost chose goal
};

RHCR_warehouse::~RHCR_warehouse() {};

void RHCR_warehouse::initialize()
{
    starts.resize(num_of_robots);
    goal_locations.resize(num_of_robots);
    paths.resize(num_of_robots);
    finished_tasks.resize(num_of_robots);
    rl_agent_poss.resize(num_of_robots);
    rl_agent_goals.resize(num_of_robots);

    timestep = 0;
    initialize_start_locations(); // randomly chose empty cell
    initialize_goal_locations();  // assign goal from eject(random) and induct station(according to cost)
    // initialize induct station counter
    for (int k = 0; k < num_of_robots; k++)
    {
        // goals
        int goal = goal_locations[k].back().first; // location of the goal
        if (G->types[goal] != "Eject" && G->types[goal] != "Induct") // at the beginning we only can only choose eject as goal
        {
            std::cout << "ERROR in the type of goal locations" << std::endl;
            std::cout << "The fiducial type of the goal of agent " << k << " at "<<goal <<" is " << G->types[goal] << std::endl;
            exit(-1);
        }
    }
    tie_breaker = solver->initialize(starts);
}

// random choose empty position to place robot
void RHCR_warehouse::initialize_start_locations()
{
    int N = G->size();
    std::vector<bool> used(N, false);
    for (int k = 0; k < num_of_robots;)
    {
        int loc = rand() % N;
        if (G->types[loc] != "Obstacle" && !used[loc])
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
void RHCR_warehouse::initialize_goal_locations()
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

// add new tasks to agents if needed for the next planning horizon
void RHCR_warehouse::update_goal_locations()
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
        int min_timesteps = G->get_Manhattan_distance(curr.first, goal.first); // cannot use h values, because graph edges may have weights
        min_timesteps = max(min_timesteps, goal.second);                      // what does second mean->the earlest arrive time
        // when the agent might finish its tasks during the next planning horizon
        // assign new tasks to the agent that keep the agent busy
        while (min_timesteps <= simulation_window)
        {
            // assign a new task, switch between induct and eject
            int next;
            if (G->types[goal.first] == "Induct") // warehouse task, need to switch between induct and eject
            {
                next = assign_eject_station();
                
            }
            else if (G->types[goal.first] == "Eject")
            {
                next = assign_induct_station(curr.first);
                drives_in_induct_stations[next]++; // number of tasks on it +++, used to calculate cost during assign new induct goal              
            }
            else
            {
                std::cout << "ERROR in update_goal_function()" << std::endl;
                std::cout << "The fiducial type should not be " << G->types[curr.first] << std::endl;
                exit(-1);
            }
            goal_locations[k].emplace_back(next, 0);
            min_timesteps += G->get_Manhattan_distance(next, goal.first); // G->heuristics.at(next)[goal];
            min_timesteps = max(min_timesteps, goal.second);
            goal = make_pair(next, 0);
        }
        for (auto pos : goal_locations[k])
            rl_agent_goals[k].emplace_back(int2Pos(pos.first));
    }
}

// call from python gym/global_reset
// update the start & goal locations of agents
void RHCR_warehouse::update_start_goal(int rl_simulation_window)
{
    simulation_window = rl_simulation_window;
    update_start_locations(); // use paths defined in move to update start
    update_goal_locations();  // generate new goal+ tansfer into py version
    solver->update(goal_locations);
}

int RHCR_warehouse::assign_induct_station(int curr) const
{
    int assigned_loc;
    auto min_cost = DBL_MAX;
    for (auto induct : drives_in_induct_stations)
    {
        double cost = G->heuristics.at(induct.first)[curr] + c * induct.second;
        if (cost < min_cost)
        {
            min_cost = cost;
            assigned_loc = induct.first;
        }
    }
    return assigned_loc;
}

// assign eject station to agent, eject station and location is randomly chosen
int RHCR_warehouse::assign_eject_station() const
{
    int n = rand() % G->ejects.size();
    boost::unordered_map<int, std::list<int>>::const_iterator it = G->ejects.begin();
    std::advance(it, n);                // 访问G.ejects容器中第n个元素（从0开始计数）。然而，由于boost::unordered_map的无序性，"第n个元素"这一说法并没有明确的顺序含义，它仅仅表示按照容器内部顺序的第n个元素。
    int p = rand() % it->second.size(); // advance move the iterator n position
    auto it2 = it->second.begin();
    std::advance(it2, p); // choice one id(location) from the eject station
    return *it2;
}

void RHCR_warehouse::finish_task(int agent_id, int location, int timestep)
{
    if (G->types[location] == "Induct")
    {
        drives_in_induct_stations[location]--; // the drive will leave the current induct station
    }
}
void RHCR_warehouse::assign_goal(int agent,int pos)
{
    return;
}