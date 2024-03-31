cmake_minimum_required(VERSION 3.11)
project(test)

set(Python_ROOT_DIR /data/text2_3/miniconda3/envs/centerpoint/bin/python)
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(handsome hello.cpp)
