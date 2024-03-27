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

find_package(fmt REQUIRED)
find_package(PCL REQUIRED)
find_package(TBB REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Iridescence REQUIRED)

set(FAST_GICP_INCLUDE_DIR /home/koide/workspace/fast_gicp/include)

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

add_library(small_gicp SHARED
  src/small_gicp/registration/registration.cpp
)
target_include_directories(small_gicp PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${TBB_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(small_gicp
  ${PCL_LIBRARIES}
  ${TBB_LIBRARIES}
)


add_executable(odometry_benchmark
  src/small_gicp/util/benchmark_odom.cpp
  src/odometry_benchmark_pcl.cpp
  src/odometry_benchmark_fast_gicp.cpp
  src/odometry_benchmark_fast_vgicp.cpp
  src/odometry_benchmark_small_gicp.cpp
  src/odometry_benchmark_small_gicp_omp.cpp
  src/odometry_benchmark_small_vgicp_omp.cpp
  src/odometry_benchmark_small_gicp_tbb.cpp
  src/odometry_benchmark_small_vgicp_tbb.cpp
  src/odometry_benchmark_small_vgicp_model_tbb.cpp
  src/odometry_benchmark_small_gicp_tbb_flow.cpp
  src/odometry_benchmark.cpp
)
target_include_directories(odometry_benchmark PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${TBB_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Iridescence_INCLUDE_DIRS}
  ${FAST_GICP_INCLUDE_DIR}
)
target_link_libraries(odometry_benchmark
  fmt::fmt
  ${PCL_LIBRARIES}
  ${TBB_LIBRARIES}
  ${Iridescence_LIBRARIES}
)

add_executable(downsampling_benchmark
  src/downsampling_benchmark.cpp
)
target_include_directories(downsampling_benchmark PUBLIC
  include
  ${PCL_INCLUDE_DIRS}
  ${TBB_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(downsampling_benchmark
  fmt::fmt
  ${PCL_LIBRARIES}
  ${TBB_LIBRARIES}
)



##########
## Test ##
##########
option(BUILD_TESTS "Build test" ON)
if(BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip)
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)

  enable_testing()
  file(GLOB TEST_SOURCES "src/test/*.cpp")
  # Generate test target for each test source file
  foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME} PUBLIC
      include
      ${PCL_INCLUDE_DIRS}
      ${TBB_INCLUDE_DIRS}
      ${EIGEN3_INCLUDE_DIR}
    )
    target_link_libraries(${TEST_NAME}
      fmt::fmt
      GTest::gtest_main
      ${PCL_LIBRARIES}
      ${TBB_LIBRARIES}
    )

    gtest_discover_tests(${TEST_NAME})
  endforeach()
endif()
