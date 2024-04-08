cmake_minimum_required(VERSION 3.16)
project(small_gicp VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(CMAKE_POSITION_INDEPENDENT_CODE "Generate position-independent code" ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Build options
option(BUILD_HELPER "Build helper library" ON)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_PYTHON_BINDINGS "Build python bindings" OFF)
option(ENABLE_COVERAGE "Enable coverage" OFF)

# Dependency options
set(BUILD_WITH_OPENMP CACHE STRING "Build with OpenMP" "auto")
option(BUILD_WITH_TBB "Build with TBB" OFF)
option(BUILD_WITH_PCL "Build with PCL (required for benchmark and test only)" OFF)
option(BUILD_WITH_FAST_GICP "Build with fast_gicp (required for benchmark and test only)" OFF)
option(BUILD_WITH_IRIDESCENCE "Build with Iridescence (required for benchmark)" OFF)
option(BUILD_WITH_MARCH_NATIVE "Build with -march=native" OFF)

# Set mandatory dependencies of optional features
if(BUILD_TESTS OR BUILD_EXAMPLES OR BUILD_BENCHMARKS)
  find_package(fmt REQUIRED)
endif()
if(BUILD_TESTS OR BUILD_EXAMPLES)
  set(BUILD_HELPER ON CACHE BOOL "Helper library is required" FORCE)
  set(BUILD_WITH_TBB ON CACHE BOOL "TBB is required" FORCE)
  set(BUILD_WITH_PCL ON CACHE BOOL "PCL is required" FORCE)
endif()

# Eigen is the sole mandatory dependency
find_package(Eigen3 REQUIRED CONFIG)

if(BUILD_WITH_OPENMP STREQUAL "auto")
  find_package(OpenMP)
  set(BUILD_WITH_OPENMP ${OpenMP_FOUND})
elseif(BUILD_WITH_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

if (BUILD_WITH_TBB)
  find_package(TBB REQUIRED)
  add_compile_definitions(BUILD_WITH_TBB)
endif()

if (BUILD_WITH_PCL)
  find_package(PCL REQUIRED)
  add_compile_definitions(BUILD_WITH_PCL)
  if (NOT TARGET PCL::PCL)
    add_library(PCL::PCL INTERFACE IMPORTED)
    set_target_properties(PCL::PCL PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${PCL_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${PCL_LIBRARIES}"
    )
  endif()
endif()

if (BUILD_WITH_IRIDESCENCE)
  find_package(Iridescence REQUIRED)
  add_compile_definitions(BUILD_WITH_IRIDESCENCE)
endif()

if (BUILD_WITH_FAST_GICP)
  if (NOT FAST_GICP_INCLUDE_DIR)
    set(FAST_GICP_INCLUDE_DIR $ENV{FAST_GICP_INCLUDE_DIR} CACHE PATH "Path to fast_gicp include directory")
  endif()
  add_compile_definitions(BUILD_WITH_FAST_GICP)
endif()

if(BUILD_WITH_MARCH_NATIVE)
  add_compile_options(-march=native)
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
    COMMAND ${LCOV} --remove coverage.info -o coverage.info '/usr/*'
    COMMAND ${GENHTML} --demangle-cpp -o coverage coverage.info
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

###########
## Build ##
###########

include_directories(include)

# Helper library
if(BUILD_HELPER)
  add_library(small_gicp
    src/small_gicp/registration/registration.cpp
    src/small_gicp/registration/registration_helper.cpp
  )
  target_include_directories(small_gicp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(small_gicp PUBLIC
    Eigen3::Eigen
    $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
  )
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(small_gicp_python src/python/python.cpp)
  set_target_properties(small_gicp_python PROPERTIES OUTPUT_NAME small_gicp)
  target_link_libraries(small_gicp_python PRIVATE small_gicp)
endif()

###############
## Benchmark ##
###############
if(BUILD_BENCHMARKS)
  # Odometry benchmark
  add_executable(odometry_benchmark
    src/small_gicp/benchmark/benchmark_odom.cpp
    src/benchmark/odometry_benchmark_pcl.cpp
    src/benchmark/odometry_benchmark_fast_gicp.cpp
    src/benchmark/odometry_benchmark_fast_vgicp.cpp
    src/benchmark/odometry_benchmark_small_gicp.cpp
    src/benchmark/odometry_benchmark_small_gicp_pcl.cpp
    src/benchmark/odometry_benchmark_small_gicp_omp.cpp
    src/benchmark/odometry_benchmark_small_vgicp_omp.cpp
    src/benchmark/odometry_benchmark_small_gicp_tbb.cpp
    src/benchmark/odometry_benchmark_small_vgicp_tbb.cpp
    src/benchmark/odometry_benchmark_small_gicp_model_tbb.cpp
    src/benchmark/odometry_benchmark_small_vgicp_model_tbb.cpp
    src/benchmark/odometry_benchmark_small_gicp_tbb_flow.cpp
    src/benchmark/odometry_benchmark.cpp
  )
  target_include_directories(odometry_benchmark PUBLIC
    ${FAST_GICP_INCLUDE_DIR}
  )
  target_link_libraries(odometry_benchmark PRIVATE
    fmt::fmt
    Eigen3::Eigen
    $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
    $<TARGET_NAME_IF_EXISTS:Iridescence::Iridescence>
    $<TARGET_NAME_IF_EXISTS:TBB::tbb>
    $<TARGET_NAME_IF_EXISTS:TBB::tbbmalloc>
    $<TARGET_NAME_IF_EXISTS:PCL::PCL>
  )

  # KdTree construction benchmark
  add_executable(kdtree_benchmark
    src/benchmark/kdtree_benchmark.cpp
  )
  target_link_libraries(kdtree_benchmark PRIVATE
    fmt::fmt
    Eigen3::Eigen
    $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
    $<TARGET_NAME_IF_EXISTS:TBB::tbb>
    $<TARGET_NAME_IF_EXISTS:TBB::tbbmalloc>
  )

  # Downsampling benchmark
  add_executable(downsampling_benchmark
    src/benchmark/downsampling_benchmark.cpp
  )
  target_link_libraries(downsampling_benchmark PRIVATE
    fmt::fmt
    Eigen3::Eigen
    $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
    $<TARGET_NAME_IF_EXISTS:TBB::tbb>
    $<TARGET_NAME_IF_EXISTS:TBB::tbbmalloc>
    $<TARGET_NAME_IF_EXISTS:PCL::PCL>
  )
endif()

#############
## Example ##
#############
if(BUILD_EXAMPLES)
  file(GLOB EXAMPLE_SOURCES "src/example/*.cpp")
  foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE
      small_gicp
      fmt::fmt
      TBB::tbb
      TBB::tbbmalloc
      PCL::PCL
    )
  endforeach()
endif()

##########
## Test ##
##########
if(BUILD_TESTS)
  find_package(GTest REQUIRED)

  enable_testing()
  file(GLOB TEST_SOURCES "src/test/*.cpp")
  # Generate test target for each test source file
  foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE
      small_gicp
      fmt::fmt
      GTest::gtest_main
      TBB::tbb
      TBB::tbbmalloc
      PCL::PCL
    )

    gtest_discover_tests(${TEST_NAME} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  endforeach()
endif()


#############
## Install ##
#############

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(BUILD_HELPER)
  install(TARGETS small_gicp
    EXPORT small_gicp-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
  set(CMAKE_CONFIG_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/small_gicp"
    CACHE PATH "Install directory for CMake config files"
  )
  include(CMakePackageConfigHelpers)
  install(EXPORT small_gicp-targets
    FILE small_gicp-targets.cmake
    NAMESPACE small_gicp::
    DESTINATION ${CMAKE_CONFIG_INSTALL_DIR}
  )
  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/small_gicp-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/small_gicp-config.cmake"
    INSTALL_DESTINATION ${CMAKE_CONFIG_INSTALL_DIR}
  )
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/small_gicp-config-version.cmake"
    VERSION ${VERSION}
    COMPATIBILITY SameMajorVersion
  )
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/small_gicp-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/small_gicp-config-version.cmake"
    DESTINATION ${CMAKE_CONFIG_INSTALL_DIR}
  )
  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTBB.cmake"
    DESTINATION ${CMAKE_CONFIG_INSTALL_DIR}
  )
endif()

if(BUILD_PYTHON_BINDINGS)
  if(DEFINED SKBUILD_PROJECT_NAME)
    set(PYTHON_INSTALL_DIR .)
  elseif(NOT DEFINED PYTHON_INSTALL_DIR)
    set(PYTHON_INSTALL_DIR lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages)
  endif()
  install(TARGETS small_gicp_python small_gicp
    LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}
    COMPONENT python
  )
endif()
