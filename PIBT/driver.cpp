#include "./pybind11/include/pybind11/pybind11.h"
#include "./pybind11/include/pybind11/stl.h"
#include "./src/BasicGraph.cpp"
#include "./src/common.cpp"
#include "./src/States.cpp"
#include "pibt.h"
#include "./src/pibt.cpp"
#include "MazeGraph.h"
#include "./src/MazeGraph.cpp"

namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(pibt_1, m)
{
    py::class_<MazeGraph>(m, "MazeGraph")
        .def(py::init<vector<vector<int>>&>())
        .def("load_map", &MazeGraph::load_map)
        .def("preprocessing", &MazeGraph::preprocessing)
        .def("visualize_heuristics_table", &MazeGraph::visualize_heuristics_table)
        .def("compute_heuristics", &MazeGraph::compute_heuristics);
    py::class_<PIBT>(m, "PIBT")
        .def(py::init<vector<vector<int>>&, int, int>())
        .def("run", &PIBT::run)
        .def("initialize", &PIBT::initialize)
        .def("update_goal", &PIBT::update_goal)
        .def("get_heuri_map", &PIBT::get_heuri_map)
        .def("agent_poss", &PIBT::agent_poss)
        .def("elapsed", &PIBT::elapsed)
        .def_readwrite("tie_breaker", &PIBT::tie_breaker)
        .def_readwrite("agent_goals", &PIBT::agent_goals);
}
