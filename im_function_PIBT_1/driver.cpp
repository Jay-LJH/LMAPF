#include "./pybind11/include/pybind11/pybind11.h"
#include "./pybind11/include/pybind11/stl.h"
#include "rhcr_class.h"
#include "./src/BasicGraph.cpp"
#include "./src/common.cpp"
#include "./src/rhcr_class.cpp"
#include "./src/SortingGraph.cpp"
#include "./src/States.cpp"
#include "./src/pibt_mapd.cpp"
#include "MazeGraph.h"
#include "./src/MazeGraph.cpp"
#include "rhcr_warehouse.h"
#include "./src/rhcr_warehouse.cpp"
#include "rhcr_maze.h"
#include "./src/rhcr_maze.cpp"

namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(lifelong_pibt_1, m)
{
    py::class_<RHCR_class_pibt_learn>(m, "RHCR_class_pibt_learn")
        .def(py::init<int, int, int, int, int, std::string>())
        .def("update_start_goal", &RHCR_class_pibt_learn::update_start_goal)
        .def("get_heuri_map", &RHCR_class_pibt_learn::get_heuri_map)
        .def("run_pibt",  py::overload_cast<>(&RHCR_class_pibt_learn::run_pibt))
        .def("run_pibt",  py::overload_cast<const vector<vector<int>> &>(&RHCR_class_pibt_learn::run_pibt))
        .def("update_system", &RHCR_class_pibt_learn::update_system)
        .def_readwrite("rl_agent_poss", &RHCR_class_pibt_learn::rl_agent_poss)
        .def_readwrite("rl_agent_goals", &RHCR_class_pibt_learn::rl_agent_goals)
        .def_readwrite("rl_path", &RHCR_class_pibt_learn::rl_path)
        .def_readwrite("num_of_tasks", &RHCR_class_pibt_learn::num_of_tasks)
        .def_readwrite("tie_breaker", &RHCR_class_pibt_learn::tie_breaker)
        .def_readwrite("timestep", &RHCR_class_pibt_learn::timestep)
        .def("print", &RHCR_class_pibt_learn::print);

    py::class_<RHCR_warehouse, RHCR_class_pibt_learn>(m, "RHCR_warehouse")
        .def(py::init<int, int, int, int, int, vector<vector<int>>, vector<vector<int>>, std::string>())
        .def("update_start_goal", &RHCR_warehouse::update_start_goal)
        .def("get_heuri_map", &RHCR_warehouse::get_heuri_map)
        .def("run_pibt",  py::overload_cast<>(&RHCR_warehouse::run_pibt))
        .def("run_pibt",  py::overload_cast<const vector<vector<int>> &>(&RHCR_warehouse::run_pibt))
        .def("update_system", &RHCR_warehouse::update_system)
        .def("assign_goal", &RHCR_warehouse::assign_goal)
        .def("finish_task", &RHCR_warehouse::finish_task)
        .def_readwrite("rl_agent_poss", &RHCR_warehouse::rl_agent_poss)
        .def_readwrite("rl_agent_goals", &RHCR_warehouse::rl_agent_goals)
        .def_readwrite("rl_path", &RHCR_warehouse::rl_path)
        .def_readwrite("num_of_tasks", &RHCR_warehouse::num_of_tasks)
        .def_readwrite("tie_breaker", &RHCR_warehouse::tie_breaker)
        .def_readwrite("timestep", &RHCR_warehouse::timestep);

    py::class_<RHCR_maze, RHCR_class_pibt_learn>(m, "RHCR_maze")
        .def(py::init<int, int,  int, vector<vector<int>>, std::string>())
        .def("update_start_goal", &RHCR_maze::update_start_goal)
        .def("get_heuri_map", &RHCR_maze::get_heuri_map)
        .def("run_pibt",  py::overload_cast<>(&RHCR_maze::run_pibt))
        .def("run_pibt",  py::overload_cast<const vector<vector<int>> &>(&RHCR_maze::run_pibt))
        .def("update_system", &RHCR_maze::update_system)
        .def_readwrite("rl_agent_poss", &RHCR_maze::rl_agent_poss)
        .def_readwrite("rl_agent_goals", &RHCR_maze::rl_agent_goals)
        .def_readwrite("rl_path", &RHCR_maze::rl_path)
        .def_readwrite("num_of_tasks", &RHCR_maze::num_of_tasks)
        .def_readwrite("tie_breaker", &RHCR_maze::tie_breaker)
        .def_readwrite("timestep", &RHCR_maze::timestep);


    py::class_<MazeGraph>(m, "MazeGraph")
        .def(py::init<vector<vector<int>>&,string,int>())
        .def("load_map", &MazeGraph::load_map)
        .def("preprocessing", &MazeGraph::preprocessing)
        .def("visualize_heuristics_table", &MazeGraph::visualize_heuristics_table)
        .def("compute_heuristics", &MazeGraph::compute_heuristics);
}
