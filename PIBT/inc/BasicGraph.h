#pragma once  //防止同一个文件在单个编译中被包含（include）多次。如果在后续的编译过程中再次遇到对同一文件的包含请求，编译器会识别出该文件已经被包含过，并忽略掉这一请求。这样，即使文件在不同的地方被多次#include，实际上它只会被编译一次，从而避免了重复定义和声明的问题。
#include "common.h"
#include "States.h"

#define WEIGHT_MAX INT_MAX/2  //assign value to weight_max as 整型（int）可以表示的最大值除以2。
enum class Type
{
    Obstacle,
    Travel,
    Human
};


class BasicGraph
{
public:
    vector<Type> types;  // type of vertex
    unordered_map<int, vector<double>> heuristics; //heuristics value from positon(key) to all others positions(value)
    list<State> get_neighbors(const State& v) const;
    list<State> get_reverse_neighbors(const State& v) const; // ignore time and wait action
    double get_weight(int from, int to) const; // fiducials from and to are neighbors   get edge weight
    int size() const { return rows * cols; }
    int get_Manhattan_distance(int loc1, int loc2) const;
    vector<int> move;  //init when SortingGraph loading map, move[0]=1, move[1]=-cols, move[2]=-1, move[3]=cols,east,north,west,south
    int get_direction(int from, int to) const;
    vector<State> get_neighbors_v(const State& s) const;
	vector<double> compute_heuristics(int root_location); // compute distances from all locations to the root location by Djstart alg

    int rows;
    int cols;
    vector<vector<double> > weights; // weight between neighbors for each position
    void save_heuristics_table(std::string fname);
    bool load_heuristics_table(std::ifstream& myfile);
    vector<vector<int>> visualize_heuristics_table(int x, int y);
};
