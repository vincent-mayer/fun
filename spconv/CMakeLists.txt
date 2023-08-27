cmake_minimum_required(VERSION 3.21)
project(spconv)

set(CMAKE_CXX_STANDARD 20)

# Set the C++ compiler to Clang
set(CMAKE_CXX_COMPILER "clang++")

# Add the source file(s) to the project
set(SOURCES spconv.cpp)

# Create the executable
add_executable(spconv ${SOURCES})