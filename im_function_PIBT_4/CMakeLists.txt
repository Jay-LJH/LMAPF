cmake_minimum_required (VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)
project(lifelong_pibt_4)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "RELEASE")
ENDIF()

include_directories("inc")

# Find Boost
find_package( Boost 1.49.0 REQUIRED COMPONENTS program_options system filesystem)
include_directories( ${Boost_INCLUDE_DIRS} )

add_subdirectory(pybind11)
pybind11_add_module(lifelong_pibt_4 driver.cpp)

target_link_libraries(lifelong_pibt_4 PRIVATE ${Boost_LIBRARIES})







