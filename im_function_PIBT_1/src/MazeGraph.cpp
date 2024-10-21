#include "MazeGraph.h"
#include <fstream>
#include "StateTimeAStar.h"
#include <sstream>
#include <random>
#include <chrono>

MazeGraph::MazeGraph(vector<vector<int>> &py_map, string path, int env_id)
{
    this->path = path;
    this->env_id = env_id;
    load_map(py_map, py_map.size(), py_map[0].size());
    preprocessing(path, env_id);
}
// initialize the maze graph from python map
// the map is a 2D array, -1 is obstacle, 0 is travel
void MazeGraph::load_map(vector<vector<int>> &py_map, int num_rows, int num_cols)
{
    this->rows = num_rows;
    this->cols = num_cols;
    this->move = {1, -cols, -1, cols};
    this->types.resize(rows * cols);
    this->weights.resize(rows * cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int pos = i * cols + j;
            if (py_map[i][j] == -1)
            {
                this->types[pos] = "Obstacle";
                obstacles.insert(pos);
            }
            else
            {
                this->types[pos] = "Travel";
                travels.insert(pos);
            }
            this->weights[pos].resize(4, 1.0);
            if (this->types[pos] == "Obstacle")
            {
                for (int k = 0; k < 4; k++)
                {
                    this->weights[pos][k] = WEIGHT_MAX;
                }
            }
            if (i == 0)
            {
                this->weights[pos][1] = WEIGHT_MAX;
            }
            if (j == 0)
            {
                this->weights[pos][2] = WEIGHT_MAX;
            }
            if (i == rows - 1)
            {
                this->weights[pos][3] = WEIGHT_MAX;
            }
            if (j == cols - 1)
            {
                this->weights[pos][0] = WEIGHT_MAX;
            }
        }
    }
}

// compute heuristics table for all nodes with cost o(n^2)
// it might be faster than read from file
void MazeGraph::preprocessing(const std::string &project_path, int env_id)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int pos = i * cols + j;
            if (this->types[pos] == "Travel")
            {
                heuristics[pos] = compute_heuristics(pos);
            }
        }
    }
}

// find a free position in the map
int MazeGraph::get_random_travel()
{
    if(travels.empty())
    {
        return -1;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    int result;
    std::sample(travels.begin(), travels.end(), &result, 1, gen);
    return result;
}