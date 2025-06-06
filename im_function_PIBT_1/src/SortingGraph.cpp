#include "SortingGraph.h"
#include <fstream>
#include "StateTimeAStar.h"
#include <sstream>
#include <random>
#include <chrono>

// build sorting grid from map
// py_map: total_map from python
// station_map: station_map from python
void SortingGrid::load_map(vector<vector<int>>& py_map,vector<vector<int>>& station_map, int num_rows, int num_cols) 
{
	this->rows = num_rows; // read number of cols
	this->cols = num_cols; // read number of rows
	move = {1, -cols, -1, cols}; // set move directions
	//read tyeps, station ids and edge weights
	this->types.resize(rows * cols); //  for each node
	this->weights.resize(rows * cols);
	for (int i = 0; i < rows; i++)
    {   for (int j=0; j<cols;j++)
            {
                int poss=i*cols+j;
                if (py_map[i][j]==-3)
                {
                    this->types[poss]="Induct";
                    this->inducts[station_map[i][j]] = poss;
                }
                else if (py_map[i][j]==-2)
                {
                    this->types[poss]="Eject";
                    int id_num=station_map[i][j];
                    boost::unordered_map<int, std::list<int> >::iterator it = this->ejects.find(id_num);
                    if (it == ejects.end())
                    {
                        this->ejects[id_num] = std::list<int>();
                    }
                    this->ejects[id_num].push_back(poss); // read eject station id , record eject and its id
                }
                else if (py_map[i][j]==-1)
                    this->types[poss]="Obstacle";
                else if (py_map[i][j]==0)
                    this->types[poss]="Travel";
                // set weight between nodes
                // if it is obstacle or exceed map, set weight to max else set to 1
                weights[poss].resize(5, 1.0); // why weight is different?
                if (this->types[poss]=="Obstacle")
                {
                    for (int k = 0; k < 4; k++) // read edge weights, currently is same weight
                        weights[poss][k] = WEIGHT_MAX;
                }
                if (i==0)
                    weights[poss][1]=WEIGHT_MAX; //north
                if (j==0)
                    weights[poss][2]=WEIGHT_MAX; //west
                if (i==rows-1)
                    weights[poss][3]=WEIGHT_MAX; //south
                if (j==cols-1)
                    weights[poss][0]=WEIGHT_MAX; //east

            }
    }
}

void SortingGrid::preprocessing(const std::string &project_path, int env_id)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int pos = i * cols + j;
            if (this->types[pos] != "Obstacle")
            {
                heuristics[pos] = compute_heuristics(pos);
            }
        }
    }
}