#pragma once
#include "common.h"

struct State //subset of path
{
    int location;
    int timestep;
    int orientation;

    struct Hasher
    {
        std::size_t operator()(const State& n) const  // overwrite
        {
            size_t loc_hash = std::hash<int>()(n.location);  // size_t = int for length
            size_t time_hash = std::hash<int>()(n.timestep); //计算某个整数（n.location）的哈希值，并将这个哈希值存储在size_t类型的变量loc_hash中
            size_t ori_hash = std::hash<int>()(n.orientation);  //std::hash<int>()创建了一个std::hash的实例，这个实例专门用于计算int类型数据的哈希值。圆括号()表示调用std::hash<int>的构造函数，以创建一个临时的std::hash<int>对象。
            return (time_hash ^ (loc_hash << 1) ^ (ori_hash << 2)); //左移位操作符
        }
    };

    void operator = (const State& other)
    {
        timestep = other.timestep;
        location = other.location;
        orientation = other.orientation;
    }

    bool operator == (const State& other) const
    {
        return timestep == other.timestep && location == other.location && orientation == other.orientation;
    }

    bool operator != (const State& other) const
    {
        return timestep != other.timestep || location != other.location || orientation != other.orientation;
    }

    State(): location(-1), timestep(-1), orientation(-1) {}
    State(int location, int timestep = -1, int orientation = -1):
            location(location), timestep(timestep), orientation(orientation) {}
    State(const State& other) {location = other.location; timestep = other.timestep; orientation = other.orientation; }
};

std::ostream & operator << (std::ostream &out, const State &s);


typedef std::vector<State> Path;

std::ostream & operator << (std::ostream &out, const Path &path);