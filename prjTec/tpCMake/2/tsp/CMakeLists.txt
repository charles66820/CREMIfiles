cmake_minimum_required(VERSION 2.6)

project(tsp)
set(CMAKE_C_FLAGS "-g -Wall -std=c99")

# solve-TSP
add_executable(solve-TSP solve-TSP.c)
target_link_libraries(solve-TSP tsp)

# solve-TSP
add_executable(convert-2D-to-full convert-2D-to-full.c)
target_link_libraries(convert-2D-to-full tsp)

# libtsp.a
add_library(tsp tsp.c memoire.c matrice2d.c)
target_link_libraries(tsp m)
