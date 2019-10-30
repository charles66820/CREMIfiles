cmake_minimum_required(VERSION 2.6)

project (Hello-World)

set(CMAKE_C_FLAGS "-std=c99 -g -Wall")

add_executable(helloworld main.c bar.c foo.c)
