cmake_minimum_required(VERSION 3.5.2)
project(small_gicp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Eigen is the sole mandatory dependency
find_package(Eigen3 REQUIRED)
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

option(BUILD_HELPER "Build helper library" ON)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_WITH_TBB "Build with TBB" ON)
option(BUILD_WITH_PCL "Build with PCL (required for benchmark and test only)" OFF)
option(BUILD_WITH_FAST_GICP "Build with fast_gicp (required for benchmark and test only)" OFF)
option(BUILD_WITH_IRIDESCENCE "Build with Iridescence (required for benchmark)" OFF)
option(BUILD_WITH_MARCH_NATIVE "Build with -march=native" OFF)
option(ENABLE_COVERAGE "Enable coverage" OFF)
option(BUILD_PYTHON_BINDINGS "Build python bindings" OFF)

if(BUILD_WITH_MARCH_NATIVE)
  add_compile_options(-march=native)
  set(CMAKE_C_FLAGS "-march=native ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}")
endif()

##############
## Coverage ##
##############
if(ENABLE_COVERAGE)
  # https://danielsieger.com/blog/2022/03/06/code-coverage-for-cpp.html
  set(CMAKE_CXX_FLAGS "-O0 -coverage")

  find_program(LCOV lcov REQUIRED)
  find_program(GENHTML genhtml REQUIRED)

  add_custom_target(coverage
    COMMAND ${LCOV} --directory . --capture --output-file coverage.info
    COMMAND ${LCOV} --remove coverage.info -o filtered_coverage.info '/usr/*'
    COMMAND ${GENHTML} --demangle-cpp -o coverage filtered_coverage.info
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

#############
## Install ##
#############
install(DIRECTORY include/ DESTINATION include)

###########
## Build ##
###########

# Helper library
if(BUILD_HELPER)
  add_library(small_gicp_helper SHARED
    src/small_gicp/registration/registration.cpp
    src/small_gicp/registration/registration_helper.cpp
  )
  target_include_directories(small_gicp_helper PUBLIC
    include
    ${EIGEN3_INCLUDE_DIR}
  )
  target_link_libraries(small_gicp_helper
    OpenMP::OpenMP_CXX
  )

  install(TARGETS small_gicp_helper DESTINATION lib)
endif()

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
  pybind11_add_module(small_gicp src/python/python.cpp)
  target_include_directories(small_gicp PUBLIC
    include
    ${EIGEN3_INCLUDE_DIR}
  )
  target_link_libraries(small_gicp PRIVATE
    small_gicp_helper
    OpenMP::OpenMP_CXX
  )
endif()

###############
## Benchmark ##
###############
if(BUILD_BENCHMARKS)
  find_package(fmt REQUIRED)
  if (BUILD_WITH_TBB)
    find_package(TBB REQUIRED)
    add_compile_definitions(BUILD_WITH_TBB)
  endif()
  if (BUILD_WITH_PCL)
    find_package(PCL REQUIRED)
    add_compile_definitions(BUILD_WITH_PCL)
  endif()
  if (BUILD_WITH_IRIDESCENCE)
    find_package(Iridescence REQUIRED)
    add_compile_definitions(BUILD_WITH_IRIDESCENCE)
  endif()
  if (BUILD_WITH_FAST_GICP)
    # set(FAST_GICP_INCLUDE_DIR /home/koide/workspace/fast_gicp/include)
    set(FAST_GICP_INCLUDE_DIR $ENV{FAST_GICP_INCLUDE_DIR})
    add_compile_definitions(BUILD_WITH_FAST_GICP)
  endif()

  # Odometry benchmark
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
    OpenMP::OpenMP_CXX
    ${PCL_LIBRARIES}
    ${TBB_LIBRARIES}
    ${Iridescence_LIBRARIES}
  )

  # KdTree construction benchmark
  add_executable(kdtree_benchmark
    src/kdtree_benchmark.cpp
  )
  target_include_directories(kdtree_benchmark PUBLIC
    include
    ${TBB_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
  )
  target_link_libraries(kdtree_benchmark
    fmt::fmt
    OpenMP::OpenMP_CXX
    ${TBB_LIBRARIES}
  )

  if(BUILD_WITH_PCL)
    # Downsampling benchmark
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
      OpenMP::OpenMP_CXX
      ${PCL_LIBRARIES}
      ${TBB_LIBRARIES}
    )
  endif()
endif()

#############
## Example ##
#############
if(BUILD_EXAMPLES)
  find_package(fmt REQUIRED)
  find_package(PCL REQUIRED)
  find_package(TBB REQUIRED)

  file(GLOB EXAMPLE_SOURCES "src/example/*.cpp")
  foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    target_include_directories(${EXAMPLE_NAME} PUBLIC
      include
      ${PCL_INCLUDE_DIRS}
      ${TBB_INCLUDE_DIRS}
      ${EIGEN3_INCLUDE_DIR}
    )
    target_link_libraries(${EXAMPLE_NAME}
      small_gicp_helper
      fmt::fmt
      OpenMP::OpenMP_CXX
      ${PCL_LIBRARIES}
      ${TBB_LIBRARIES}
    )
  endforeach()
endif()

##########
## Test ##
##########
if(BUILD_TESTS)
  find_package(fmt REQUIRED)
  find_package(PCL REQUIRED)
  find_package(TBB REQUIRED)
  find_package(GTest REQUIRED)

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
      small_gicp_helper
      fmt::fmt
      OpenMP::OpenMP_CXX
      GTest::gtest_main
      ${PCL_LIBRARIES}
      ${TBB_LIBRARIES}
    )

    gtest_discover_tests(${TEST_NAME} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  endforeach()
endif()
