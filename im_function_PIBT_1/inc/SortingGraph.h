#pragma once
#include "BasicGraph.h"


class SortingGrid : // static map
	public BasicGraph
{
public:
    unordered_map<int, int> inducts;   // station ->id
    unordered_map<int, list<int> > ejects; // one eject station could have multiple eject fiducials  // station->id list

    void load_map(vector<vector<int>>& py_map,vector<vector<int>>& station_map, int num_rows, int num_cols);
    void preprocessing(const std::string& project_path,int env_id); // compute heuristics
};
