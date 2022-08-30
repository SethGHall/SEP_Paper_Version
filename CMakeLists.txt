
cmake_minimum_required(VERSION 3.10.2)

#! Base project
project(imaging C CXX CUDA)

# Add the cmake folder so the FindSphinx module is found
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

add_subdirectory("imaging")

if("${CMAKE_BUILD_TYPE}" STREQUAL "DOCS")
	add_subdirectory("docs")
endif()

if ("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_C_FLAGS "-Wall -Wextra")
	set(CMAKE_C_FLAGS_DEBUG "-g -O0")
	set(CMAKE_C_FLAGS_RELEASE "-O3")
	set(CMAKE_CXX_FLAGS "-Wall -Wextra")
	set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()
