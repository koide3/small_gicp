find_path(TBB_INCLUDE_DIRS tbb/tbb.h
  HINTS /usr/local/include /usr/include
  DOC "oneTBB include directories")

find_library(TBB_LIB NAMES tbb
  HINTS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
  DOC "TBB libraries")

# if(GTSAM_LIB AND GTSAM_UNSTABLE_LIB AND TBB_LIB)
if(TBB_LIB)
  set(TBB_LIBRARIES ${TBB_LIB})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB DEFAULT_MSG TBB_INCLUDE_DIRS TBB_LIBRARIES)
