cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# specifically for docker env 
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.2)

set(PROJ_NAME "gpgtpu")
project(${PROJ_NAME} VERSION 1.0 LANGUAGES CXX CUDA)
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_C_FLAGS "-pthread -std=c++17")
set(CMAKE_C_FLAGS_DEBUG "-g -O4")
set(CMAKE_C_FLAGS_RELEASE "-O4")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
set(CMAKE_CUDA_STANDARD 17)

# Add a series of utils libraries
add_subdirectory(utils)
add_subdirectory(src/kernels)

# Find external packages
find_package(OpenCV REQUIRED)
find_package(CUDA 10.2 REQUIRED)
OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
ENDIF()
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(./utils/include) # for Common.h 
include_directories(./utils/common/inc) # for helper_cuda.h 

# Find GPTPU library
find_library(GPTPU_UTILS_LIB gptpu_utils /usr/local/)
message(STATUS ${GPTPU_UTILS_LIB})

file(GLOB SRC_FILES src/*.cpp)
add_executable(${PROJ_NAME} ${SRC_FILES})
target_include_directories(${PROJ_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJ_NAME} PUBLIC utils kernels ${OpenCV_LIBS} ${GPTPU_UTILS_LIB})
