#include "SortingGraph.h"
#include <fstream>
#include "StateTimeAStar.h"
#include <sstream>
#include <random>
#include <chrono>

void SortingGrid::load_map(vector<vector<int>>& py_map,vector<vector<int>>& station_map, int num_rows, int num_cols)  //type,station,weight?  -- need to reread paper to understand what the situation mean
{
	this->rows = num_rows; // read number of cols
	this->cols = num_cols; // read number of rows
	move[0] = 1;
	move[1] = -cols;
	move[2] = -1;
	move[3] = cols;
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
                    boost::unordered_map<int, std::list<int> >::iterator it = ejects.find(id_num);
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


void SortingGrid::preprocessing(const std::string& project_path,int env_id)
{
    std::string fname;
    fname = project_path+"/"+std::to_string(env_id) +std::to_string(rows) + std::to_string(cols)+ "_heuristics_table.txt";
    std::ifstream myfile(fname.c_str());
    if (myfile.is_open())
    {
        bool succ = load_heuristics_table(myfile);
        myfile.close();
        if (!succ)
        {
            std::cout << "Error: Heuristics table is not correct!" << std::endl;
            exit(1);
        }
        // ensure that the heuristic table is correct
    }
    else
    {
        for (auto induct : inducts)
        {
            heuristics[induct.second] = compute_heuristics(induct.second);  // first is id, second is location
        }
        for (auto eject_station : ejects)
        {
            for (int eject : eject_station.second)
            {
                heuristics[eject] = compute_heuristics(eject);
            }
        }
        save_heuristics_table(fname);
    }

}
