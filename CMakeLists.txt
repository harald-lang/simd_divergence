cmake_minimum_required(VERSION 3.5.1)
project(simd_divergence)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -march=knl -O3 -Wall -Wno-deprecated -Wno-unused-variable")

set(SOURCE_FILES
        src/refill.hpp
        src/util.hpp
        )

add_executable(refill_test src/refill_test.cpp ${SOURCE_FILES})

