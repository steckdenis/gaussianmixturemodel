cmake_minimum_required(VERSION 2.8)

project(gaussianmixture)
add_definitions(-std=c++11)

find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 eigen3 REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)

# Library install path
get_property(LIB64 GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS)

if ("${LIB64}" STREQUAL "TRUE")
    set(LIBSUFFIX 64)
else()
    set(LIBSUFFIX "")
endif()

# Library
add_library(gaussianmixturemodel SHARED
    gaussianmixture.cpp
    gaussianmixture_c.cpp
)
set_target_properties(gaussianmixturemodel PROPERTIES
    VERSION 1
    SOVERSION 1
)

install(
    TARGETS gaussianmixturemodel
    DESTINATION lib${LIBSUFFIX}
)
install(
    FILES gaussianmixture.h
    DESTINATION include
)