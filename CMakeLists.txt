cmake_minimum_required(VERSION 2.8)

project(gaussianmixture)
add_definitions(-std=c++11)

find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 eigen3 REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)

# Generic library
add_library(gaussianmixturemodel SHARED
    gaussianmixture.cpp
    gaussianmixture_c.cpp
)
