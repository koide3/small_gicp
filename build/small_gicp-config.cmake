# Config file for the small_gicp package
#
# Usage from an external project:
#
#  find_package(small_gicp REQUIRED)
#  target_link_libraries(MY_TARGET_NAME small_gicp::small_gicp)
#
# Optionally, for TBB support in *_tbb.hpp headers also add:
#
#  find_package(TBB REQUIRED)
#  target_link_libraries(MY_TARGET_NAME TBB::tbb TBB::tbbmalloc)
#

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was small_gicp-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include_guard()

set(BUILD_WITH_OPENMP TRUE)

include(CMakeFindDependencyMacro)
find_dependency(Eigen3 REQUIRED)
if (BUILD_WITH_OPENMP)
  find_dependency(OpenMP REQUIRED COMPONENTS CXX)
endif()

# For FindTBB.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

include("${CMAKE_CURRENT_LIST_DIR}/small_gicp-targets.cmake")
