cmake_minimum_required(VERSION 3.15)
project(pheromone_cpp_backend VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -pthread")

# Check for AVX2 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    message(STATUS "AVX2 support enabled")
else()
    message(WARNING "AVX2 not supported, SIMD optimizations will be limited")
endif()

# Find dependencies
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG)
find_package(Boost REQUIRED)

# If pybind11 not found via config, try to find it as a Python package
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(pybind11_DIR)
        find_package(pybind11 CONFIG PATHS ${pybind11_DIR})
    endif()
endif()

if(NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
endif()

# Try to find nlohmann_json
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found via package manager, using header-only fallback")
    # Download nlohmann/json header if not available
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/nlohmann/json.hpp")
        file(DOWNLOAD
            "https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/nlohmann/json.hpp"
            SHOW_PROGRESS
        )
    endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${Python_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# Source files
# Phase 1: Communication Codec (completed)
# Phase 2: Field Operations (SIMD-optimized pheromone operations)
# Phase 3: Spatial Indexing (R-tree for efficient spatial queries)
set(SOURCES
    src/message_codec.cpp
    src/thread_pool.cpp
    src/field_operations.cpp
    src/spatial_index.cpp
)

# Create Python module
pybind11_add_module(cpp_accelerators bindings/pybind_module.cpp ${SOURCES})

if(nlohmann_json_FOUND)
    target_link_libraries(cpp_accelerators PRIVATE nlohmann_json::nlohmann_json)
endif()

# Link Boost libraries
target_link_libraries(cpp_accelerators PRIVATE ${Boost_LIBRARIES})

# Set RPATH for better library finding
set_target_properties(cpp_accelerators PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install to src/core directory for easy Python import
install(TARGETS cpp_accelerators
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/../src/core
)

message(STATUS "==============================================")
message(STATUS "C++ Backend Configuration Summary")
message(STATUS "==============================================")
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "pybind11 found: ${pybind11_FOUND}")
message(STATUS "AVX2 support: ${COMPILER_SUPPORTS_AVX2}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "==============================================")
