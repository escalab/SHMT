file(GLOB SRC "*.cpp")
add_library(utils ${SRC})

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
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
include_directories(./common/inc) # for helper_cuda.h

target_include_directories(utils PUBLIC include)
target_link_libraries(utils PUBLIC ${OpenCV_LIBS})
