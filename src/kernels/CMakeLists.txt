file(GLOB SRC "*.cpp" "*.cu")
add_library(kernels ${SRC})

list(APPEND CUDA_DEV_LIBRARIES
            ${CUDA_cufft_LIBRARY})

# Jetson nano specific flag to ensure optimal warmup performance
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -gencode arch=compute_53,code=sm_53
                       >)

find_package(OpenCV REQUIRED)
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
include_directories(./../../utils/common/inc) # for helper_cuda.h
target_include_directories(kernels PUBLIC include ../../include ../../utils/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(kernels PUBLIC utils ${OpenCV_LIBS} ${CUDA_DEV_LIBRARIES})
