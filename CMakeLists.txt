cmake_minimum_required(VERSION 3.5.2)
project(small_gicp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(BUILD_TESTS "Build test" OFF)
option(BUILD_WITH_MARCH_NATIVE "Build with -march=native" OFF)

if(BUILD_WITH_MARCH_NATIVE)
  add_compile_options(-march=native)
  set(CMAKE_C_FLAGS "-march=native ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}")
endif()

find_package(PCL REQUIRED)
find_package(TBB REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Iridescence REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

###########
## Build ##
###########

# add_executable(small_gicp_test
#   src/small_gicp_test.cpp
# )
# target_include_directories(small_gicp_test PUBLIC
#   include
#   ${PCL_INCLUDE_DIRS}
#   ${TBB_INCLUDE_DIRS}
#   ${EIGEN3_INCLUDE_DIR}
#   ${Iridescence_INCLUDE_DIRS}
# )
# target_link_libraries(small_gicp_test
#   ${PCL_LIBRARIES}
#   ${TBB_LIBRARIES}
#   ${Iridescence_LIBRARIES}
# )

add_executable(downsampling_benchmark
  src/downsampling_benchmark.cpp
)
target_include_directories(downsampling_benchmark PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${TBB_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Iridescence_INCLUDE_DIRS}
)
target_link_libraries(downsampling_benchmark
  ${PCL_LIBRARIES}
  ${TBB_LIBRARIES}
  ${Iridescence_LIBRARIES}
)