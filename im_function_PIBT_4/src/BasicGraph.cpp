#include "BasicGraph.h"
#include <boost/tokenizer.hpp>
#include "StateTimeAStar.h"
#include <random>


list<State> BasicGraph::get_neighbors(const State& s) const
{
    list<State> neighbors;
    if (s.location < 0)
        return neighbors;

    neighbors.push_back(State(s.location, s.timestep + 1)); // wait
    for (int i = 0; i < 4; i++) // move
        if (weights[s.location][i] < WEIGHT_MAX - 1 && s.location + move[i]>=0 && s.location + move[i]< this->size() && types[s.location + move[i]]!="Obstacle")
            neighbors.push_back(State(s.location + move[i], s.timestep + 1));
    return neighbors;
}


vector<State> BasicGraph::get_neighbors_v(const State& s) const
{
    vector<State> neighbors;
    if (s.location < 0)
        return neighbors;

    neighbors.emplace_back(s.location, s.timestep + 1); // wait
    for (int i = 0; i < 4; i++) // move
        if (weights[s.location][i] < WEIGHT_MAX - 1 && s.location + move[i]>=0 && s.location + move[i]< this->size() && types[s.location + move[i]]!="Obstacle")
            neighbors.emplace_back(s.location + move[i], s.timestep + 1);
    return neighbors;
}

std::list<State> BasicGraph::get_reverse_neighbors(const State& s) const
{
    std::list<State> rneighbors;
    // no wait actions
    for (int i = 0; i < 4; i++) // move
        if (s.location - move[i] >= 0 && s.location - move[i] < this->size() &&
                weights[s.location - move[i]][i] < WEIGHT_MAX - 1  && types[s.location - move[i]]!="Obstacle")   // does not exceed the graph size and not obdtacles
            rneighbors.push_back(State(s.location - move[i]));
    return rneighbors;
}


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



std::vector<double> BasicGraph::compute_heuristics(int root_location)  // compute distances from all locations to the root location
{   // revese node, only use g, change the stop condition form arrive goal to heap=empty
    std::vector<double> res(this->size(), DBL_MAX);  //double类型能表示的最大正数值
	fibonacci_heap< StateTimeAStarNode*, compare<StateTimeAStarNode::compare_node> > heap;  //存储的是指向StateTimeAStarNode类型对象的指针, compare-用于定义堆中元素的排序准则
    unordered_set< StateTimeAStarNode*, StateTimeAStarNode::Hasher, StateTimeAStarNode::EqNode> nodes; // type, hash value,if two node equal
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