#include "./pybind11/include/pybind11/pybind11.h"
#include "./pybind11/include/pybind11/stl.h"
// #include "rhcr_class.h"
#include "./src/BasicGraph.cpp"
#include "./src/common.cpp"
// #include "./src/rhcr_class.cpp"
// #include "./src/SortingGraph.cpp"
#include "./src/States.cpp"
#include "pibt_mapd.h"
#include "./src/pibt_mapd.cpp"
#include "MazeGraph.h"
#include "./src/MazeGraph.cpp"
// #include "rhcr_warehouse.h"
// #include "./src/rhcr_warehouse.cpp"
// #include "rhcr_maze.h"
// #include "./src/rhcr_maze.cpp"

namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(PIBT, m)
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
        .def("update_goal", &PIBT::update_goal);
}
