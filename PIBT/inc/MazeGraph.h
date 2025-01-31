#pragma once
#include "BasicGraph.h"


// load a maze map
// goal and start locations can be at any not obstacle position
class MazeGraph: 
	public BasicGraph
{
public:
    unordered_set<int> obstacles;
    unordered_set<int> travels;
    string path;
    int env_id; 
    MazeGraph* G;

    MazeGraph(vector<vector<int>>& py_map);
    void load_map(vector<vector<int>>& py_map);
    void preprocessing(); 
    int get_random_travel();
};
