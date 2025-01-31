#include "BasicGraph.h"
#include <boost/tokenizer.hpp>
#include "StateTimeAStar.h"
#include <random>

// get all 5 possible neighbors of a vertex by location
list<State> BasicGraph::get_neighbors(const State& s) const
{
    list<State> neighbors;
    if (s.location < 0)
        return neighbors;

    neighbors.push_back(State(s.location, s.timestep + 1)); // wait
    for (int i = 0; i < 4; i++) // move
        if (weights[s.location][i] < WEIGHT_MAX - 1 && s.location + move[i]>=0 
            && s.location + move[i]< this->size() && types[s.location + move[i]]!=Type::Obstacle)
            neighbors.push_back(State(s.location + move[i], s.timestep + 1));
    return neighbors;
}

// return vector of neighbors instead of list
vector<State> BasicGraph::get_neighbors_v(const State& s) const
{
    auto neighbors = get_neighbors(s);
    return vector(neighbors.begin(), neighbors.end());
}

// get all 4 possible neighbors of a vertex by location, without wait action compared to get_neighbors
std::list<State> BasicGraph::get_reverse_neighbors(const State& s) const
{
    std::list<State> rneighbors;
    // no wait actions
    for (int i = 0; i < 4; i++) // Traverse all the neighbors of the current node
        // does not exceed the graph size and not obdtacles
        if (s.location - move[i] >= 0 && s.location - move[i] < this->size() &&
                weights[s.location - move[i]][i] < WEIGHT_MAX - 1  && types[s.location - move[i]]!=Type::Obstacle)   
            rneighbors.push_back(State(s.location - move[i]));
    return rneighbors;
}

// get the weight of the edge between two positions
double BasicGraph::get_weight(int from, int to) const
{
    if (from == to) // wait or rotate
        return weights[from][4];
    int dir = get_direction(from, to);
    if (dir >= 0)
        return weights[from][dir];
    else
        return WEIGHT_MAX;
}

// get the direction of the edge between two positions
int BasicGraph::get_direction(int from, int to) const
{
    for (int i = 0; i < 4; i++)
    {
        if (move[i] == to - from)
            return i;
    }
    if (from == to)
        return 4;
    return -1;
}

// compute distances from all locations to the root location use Dijkstra
// revese node, only use g, change the stop condition form arrive goal to heap=empty
// return res[position] = heuristic val from root to position
std::vector<double> BasicGraph::compute_heuristics(int root_location)  
{  
    // initialize the distance to all locations as infinity
    std::vector<double> res(this->size(), DBL_MAX);  
    //fibonacci_heap keep the order of the node, unordered_set keep the node, pop the node with smallest g value
	fibonacci_heap< StateTimeAStarNode*, compare<StateTimeAStarNode::compare_node> > heap;  
    unordered_set< StateTimeAStarNode*, StateTimeAStarNode::Hasher, StateTimeAStarNode::EqNode> nodes;
    // heap for sorting, nodes for storing
    State root_state(root_location);

    StateTimeAStarNode* root = new StateTimeAStarNode(root_state, 0, 0, nullptr, 0);
    root->open_handle = heap.push(root);  // add root to heap
    nodes.insert(root);       // add root to hash_table (nodes)

	while (!heap.empty())
    {
        StateTimeAStarNode* curr = heap.top();
		heap.pop();
		for (auto next_state : get_reverse_neighbors(curr->state))  // from root to all other position, so we should use reverse neighbor at here
		{
			double next_g_val = curr->g_val + get_weight(next_state.location, curr->state.location);
            StateTimeAStarNode* next = new StateTimeAStarNode(next_state, next_g_val, 0, nullptr, 0);  // keep the h value=,so it is Djstart search
			auto it = nodes.find(next);
			if (it == nodes.end()) // did not find the node in the table
			{  // add the newly generated node to heap and hash table
				next->open_handle = heap.push(next);
				nodes.insert(next);
			}
			else 
			{  // update existing node's g_val if needed (only in the heap)
				delete(next);  // not needed anymore -- we already generated it before
                StateTimeAStarNode* existing_next = *it;
				if (existing_next->g_val > next_g_val) 
				{
					existing_next->g_val = next_g_val;
					heap.increase(existing_next->open_handle); //update ethe node in hash table use handle for faster speed
				}
			}
		}
	}
	// iterate over all nodes and populate the distances
	for (auto it = nodes.begin(); it != nodes.end(); it++)
	{
        StateTimeAStarNode* s = *it;
		res[s->state.location] = std::min(s->g_val, res[s->state.location]); // record the distance for all node in the nodes
		delete (s);
	}
	nodes.clear();
	heap.clear();
    return res;
}


int BasicGraph::get_Manhattan_distance(int loc1, int loc2) const
{
    return abs(loc1 / cols - loc2 / cols) + abs(loc1 % cols - loc2 % cols);
}

void BasicGraph::save_heuristics_table(std::string fname)
{
    std::ofstream myfile;
    myfile.open (fname);
    myfile << "table_size" << std::endl <<
           heuristics.size() << "," << this->size() << std::endl;
    for (auto h_values: heuristics)
    {
        myfile << h_values.first << std::endl;  // goal poss
        for (double h : h_values.second) // h for every location
        {
            myfile << h << ",";
        }
        myfile << std::endl;
    }
    myfile.close();
}

bool BasicGraph::load_heuristics_table(std::ifstream& myfile)
{
    boost::char_separator<char> sep(",");
    boost::tokenizer< boost::char_separator<char> >::iterator beg;
    std::string line;

    getline(myfile, line); //skip "table_size"
    getline(myfile, line);  //table size and world size
    boost::tokenizer< boost::char_separator<char> > tok(line, sep);
    beg = tok.begin();
    int N = atoi ( (*beg).c_str() ); // read number of cols
    beg++;
    int M = atoi ( (*beg).c_str() ); // read number of rows
    if (M != this->size())
        return false;
    for (int i = 0; i < N; i++)  // for each goal location
    {
        getline (myfile, line);  //position of goal
        int loc = atoi(line.c_str());
        getline (myfile, line);  // heuristic value of all vertex
        boost::tokenizer< boost::char_separator<char> > tok(line, sep);
        beg = tok.begin();
        std::vector<double> h_table(this->size());
        for (int j = 0; j < this->size(); j++)
        {
            h_table[j] = atof((*beg).c_str());
            beg++;
        }
        heuristics[loc] = h_table;
    }
    return true;
}

vector<vector<int>> BasicGraph::visualize_heuristics_table(int x, int y)
{
    vector<vector<int>> res(this->cols, vector<int>(this->rows, 0));
    auto h_table = heuristics[x * this->cols + y];
    for(int i=0;i<h_table.size();i++)
    {
        if(h_table[i] < DBL_MAX)
            res[i/this->cols][i%this->cols] = h_table[i];
        else
            res[i/this->cols][i%this->cols] = -1;
    }
    return res;
}