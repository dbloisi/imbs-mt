project(imbs)

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV REQUIRED)

include_directories(${PROJECT_BINARY_DIR})

# Build type configuration.
if (CMAKE_BUILD_TYPE)
	string(TOUPPER ${CMAKE_BUILD_TYPE} _build_type)
	set(CMAKE_BUILD_TYPE ${_build_type} CACHE STRING "" FORCE)
endif (CMAKE_BUILD_TYPE)

# Build type must be either DEBUG or RELEASE.
if (CMAKE_BUILD_TYPE STREQUAL "")
	message("No build type specified: forcing to RELEASE.")
	set(CMAKE_BUILD_TYPE "RELEASE" CACHE STRING "" FORCE)
endif (CMAKE_BUILD_TYPE STREQUAL "")

if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
	add_definitions(-ggdb)
elseif (CMAKE_BUILD_TYPE STREQUAL "RELEASE")
	add_definitions(-O3)
else (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
	message("Invalid build type specified: doing nothing.")
endif (CMAKE_BUILD_TYPE STREQUAL "DEBUG")

add_executable (imbs-mt imbs.cpp imbsmultithread.cpp imagemanager.cpp hungarianalg.cpp main.cpp)
target_link_libraries (imbs-mt ${OpenCV_LIBS})
