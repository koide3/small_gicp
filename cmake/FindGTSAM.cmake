find_path(GTSAM_INCLUDE_DIRS gtsam/inference/FactorGraph.h
  HINTS /usr/local/include /usr/include
  DOC "GTSAM include directories")

find_library(GTSAM_LIB NAMES gtsam
  HINTS /usr/local/lib /usr/lib
  DOC "GTSAM libraries")

find_library(GTSAM_UNSTABLE_LIB NAMES gtsam_unstable
  HINTS /usr/local/lib /usr/lib
  DOC "GTSAM_UNSTABLE libraries")

find_dependency(TBB REQUIRED)

add_library(gtsam INTERFACE IMPORTED GLOBAL)
set_target_properties(gtsam PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${GTSAM_INCLUDE_DIRS}"
  INTERFACE_LINK_LIBRARIES "${GTSAM_LIB} TBB::tbb TBB::tbbmalloc")

add_library(gtsam_unstable INTERFACE IMPORTED GLOBAL)
set_target_properties(gtsam_unstable PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${GTSAM_INCLUDE_DIRS}"
  INTERFACE_LINK_LIBRARIES "${GTSAM_UNSTABLE_LIB} gtsam")

if(GTSAM_LIB AND GTSAM_UNSTABLE_LIB AND TARGET TBB::tbb AND TARGET TBB::tbbmalloc)
  set(GTSAM_LIBRARIES gtsam gtsam_unstable)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GTSAM DEFAULT_MSG GTSAM_INCLUDE_DIRS GTSAM_LIBRARIES)
