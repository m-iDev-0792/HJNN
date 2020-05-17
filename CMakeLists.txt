cmake_minimum_required(VERSION 3.15)
project(HJNN)

set(CMAKE_CXX_STANDARD 14)
include_directories(include)
add_executable(HJNN main.cpp HJNN.cpp Layer.cpp utility.cpp)