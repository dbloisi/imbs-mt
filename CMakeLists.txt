project(imbs-mt)

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(OpenCV REQUIRED)

include_directories(${PROJECT_BINARY_DIR})

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable (${PROJECT_NAME} imbs.cpp imbsmultithread.cpp imagemanager.cpp main.cpp)
if (WIN32)
target_link_libraries (${PROJECT_NAME} ${OpenCV_LIBS})
else ()
target_link_libraries (${PROJECT_NAME} ${OpenCV_LIBS} -lm -lpthread)
endif()

