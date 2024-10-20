#pragma once
#include "BasicGraph.h"

// load a maze map
// goal and start locations can be at any not obstacle position
class MazeGraph: 
	public BasicGraph
{
public:
    unordered_set<int> map_travels;
    string path;
    int env_id; 
    MazeGraph(vector<vector<int>>& py_map,string path,int env_id);
    void load_map(vector<vector<int>>& py_map, int num_rows, int num_cols);
    void preprocessing(const std::string& project_path,int env_id);
};
