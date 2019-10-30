cmake_minimum_required(VERSION 2.6)

project(Hello-World)

set(CMAKE_C_FLAGS "-std=c99 -g -Wall")

add_executable(helloworld main.c)

target_link_libraries(helloworld foobar)

add_library(foobar bar.c foo.c)

target_link_libraries(foobar m)
