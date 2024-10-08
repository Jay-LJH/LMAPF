#include "./pybind11/include/pybind11/pybind11.h"
#include "./pybind11/include/pybind11/stl.h"
#include "rhcr_class.h"
#include "./src/BasicGraph.cpp"
#include "./src/common.cpp"
#include "./src/rhcr_class.cpp"
#include "./src/SortingGraph.cpp"
#include "./src/States.cpp"
#include "./src/pibt_mapd.cpp"

namespace py=pybind11;
using namespace std;

PYBIND11_MODULE(lifelong_pibt_1, m) {
    py::class_<RHCR_class_pibt_learn>(m, "RHCR_class_pibt_learn")
        .def(py::init<int ,int, int ,int, int,vector<vector<int>>,vector<vector<int>>,std::string>())
        .def("update_start_goal", &RHCR_class_pibt_learn::update_start_goal)
        .def("obtaion_heuri_map", &RHCR_class_pibt_learn::obtaion_heuri_map)
        .def("run_pibt", &RHCR_class_pibt_learn::run_pibt)
        .def("update_system", &RHCR_class_pibt_learn::update_system)
        .def_readwrite("rl_agent_poss", &RHCR_class_pibt_learn::rl_agent_poss)
        .def_readwrite("rl_agent_goals", &RHCR_class_pibt_learn::rl_agent_goals)
        .def_readwrite("rl_path", &RHCR_class_pibt_learn::rl_path)
        .def_readwrite("num_of_tasks", &RHCR_class_pibt_learn::num_of_tasks)
        .def_readwrite("tie_breaker", &RHCR_class_pibt_learn::tie_breaker)
        .def_readwrite("timestep", &RHCR_class_pibt_learn::timestep);}

