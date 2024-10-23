#include "rhcr_maze.h"

RHCR_maze::RHCR_maze(int seed, int num_of_robots,
                     int env_id, vector<vector<int>> py_map, std::string project_path) : RHCR_class_pibt_learn(seed, num_of_robots, py_map.size(), py_map[0].size(), env_id, project_path)
{
    RHCR_class_pibt_learn::G = new MazeGraph(py_map, project_path, env_id);
    solver = new PIBT_MAPD(*RHCR_class_pibt_learn::G, seed, rows * cols);
    this->G = static_cast<MazeGraph *>(RHCR_class_pibt_learn::G);
    this->env_id = env_id;
    this->path = project_path;
    srand(seed);
    G->load_map(py_map, rows, cols);
    G->preprocessing(project_path, env_id); // calculated heuristic of all goals
    initialize();                           // random choose start location and according to cost chose goal
}

RHCR_maze::~RHCR_maze() {};

void RHCR_maze::initialize()
{
    starts.resize(num_of_robots);
    goal_locations.resize(num_of_robots);
    paths.resize(num_of_robots);
    for (auto path : paths)
    {
        path.clear();
    }
    finished_tasks.resize(num_of_robots);
    rl_agent_poss.resize(num_of_robots);
    rl_agent_goals.resize(num_of_robots);
    goal_num.resize(rows * cols, 0);
    timestep = 0;
    initialize_start_locations(); // randomly chose empty cell
    initialize_goal_locations();  // assign goal from eject(random) and induct station(according to cost)
    tie_breaker = solver->initialize(starts);
    //print();
}

// random choose empty position to place robot
void RHCR_maze::initialize_start_locations()
{
    int N = G->size();
    std::vector<bool> used(N, false);
    for (int k = 0; k < num_of_robots; k++)
    {
        int loc = G->get_random_travel();
        while (used[loc])
        {
            loc = G->get_random_travel();
        }
        int orientation = -1;
        starts[k] = State(loc, 0, orientation);
        rl_agent_poss[k] = int2Pos(loc);
        paths[k].emplace_back(starts[k]);
        used[loc] = true;
        finished_tasks[k].emplace_back(loc, 0);
    }
}

// set goal location for each agent
void RHCR_maze::initialize_goal_locations()
{
    for (int k = 0; k < num_of_robots; k++)
    {
        int goal = G->get_random_travel();
        assign_goal(k, goal);
    }
}

// add new tasks to agents if needed for the next planning horizon
void RHCR_maze::update_goal_locations()
{
    // rl_agent_goals.clear();
    // rl_agent_goals.resize(num_of_robots);
    // process each agent
    for (int k = 0; k < num_of_robots; k++)
    {
        pair<int, int> curr(paths[k][timestep].location, timestep); // current location
        pair<int, int> goal;                                        // The last goal location & id
        if (goal_locations[k].empty())                              // all tasks have been completed, the agent should wait
        {
            goal = curr;
        }
        else
        {
            goal = goal_locations[k].back();
        }
        int min_timesteps = G->get_Manhattan_distance(curr.first, goal.first); // cannot use h values, because graph edges may have weights
        min_timesteps = max(min_timesteps, goal.second);                       // what does second mean->the earlest arrive time
        // when the agent might finish its tasks during the next planning horizon
        // assign new tasks to the agent that keep the agent busy
        while (min_timesteps <= simulation_window)
        {
            // assign a new task, switch between induct and eject
            int next = G->get_random_travel();
            assign_goal(k, next);
            min_timesteps += G->get_Manhattan_distance(next, goal.first); // G->heuristics.at(next)[goal];
            min_timesteps = max(min_timesteps, goal.second);
            goal = make_pair(next, 0);
        }
    }
}

// call from python gym/global_reset
// update the start & goal locations of agents
void RHCR_maze::update_start_goal(int rl_simulation_window)
{
    simulation_window = rl_simulation_window;
    update_start_locations(); // use paths defined in move to update start
    update_goal_locations(); // generate new goal+ tansfer into py version
    solver->update(goal_locations);
}

// assign goal to agent
// agent: agent id, pos: goal location
void RHCR_maze::assign_goal(int agent, int pos)
{
    goal_num[pos]++;
    rl_agent_goals[agent].emplace_back(int2Pos(pos));
    goal_locations[agent].emplace_back(pos, 0);
}

void RHCR_maze::finish_task(int agent_id, int location, int timestep)
{
    goal_num[location]--;
}