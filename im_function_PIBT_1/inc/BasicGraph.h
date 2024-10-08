#pragma once  //防止同一个文件在单个编译中被包含（include）多次。如果在后续的编译过程中再次遇到对同一文件的包含请求，编译器会识别出该文件已经被包含过，并忽略掉这一请求。这样，即使文件在不同的地方被多次#include，实际上它只会被编译一次，从而避免了重复定义和声明的问题。
#include "common.h"
#include "States.h"

#define WEIGHT_MAX INT_MAX/2  //assign value to weight_max as 整型（int）可以表示的最大值除以2。


class BasicGraph
{
public:
    vector<std::string> types;  // type of vertex
    unordered_map<int, vector<double>> heuristics; //harsh table 通过键来存取值的方法，其中每个键都是唯一的，而且每个键都映射到一个特定的值,容器中的元素是无序的。这与map容器不同，map中的元素是根据键排序存储的
    list<State> get_neighbors(const State& v) const;
    list<State> get_reverse_neighbors(const State& v) const; // ignore time
    double get_weight(int from, int to) const; // fiducials from and to are neighbors   get edge weight
    int size() const { return rows * cols; }
    int get_Manhattan_distance(int loc1, int loc2) const;
    int move[4];  //move=(0,1,2,3) ? defined when load map
    int get_direction(int from, int to) const;
    vector<State> get_neighbors_v(const State& s) const;
	vector<double> compute_heuristics(int root_location); // compute distances from all locations to the root location by Djstart alg

    int rows;
    int cols;
    vector<vector<double> > weights; // (directed) weighted 4-neighbor grid
    void save_heuristics_table(std::string fname);
    bool load_heuristics_table(std::ifstream& myfile);
};
