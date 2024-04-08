find_path(TBB_INCLUDE_DIRS tbb/tbb.h
  HINTS /usr/local/include /usr/include
  DOC "oneTBB include directories")

find_library(TBB_LIB NAMES tbb
  HINTS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
  DOC "TBB libraries")

find_library(TBB_MALLOC_LIB NAMES tbbmalloc
  HINTS /usr/local/lib /usr/lib
  DOC "TBB malloc libraries")

add_library(TBB::tbb INTERFACE IMPORTED GLOBAL)
set_target_properties(TBB::tbb PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIRS}"
  INTERFACE_LINK_LIBRARIES "${TBB_LIBRARIES}")

add_library(TBB::tbbmalloc INTERFACE IMPORTED GLOBAL)
set_target_properties(TBB::tbbmalloc PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIRS}"
  INTERFACE_LINK_LIBRARIES "${TBB_LIBRARIES}")

if(TBB_LIB AND TBB_MALLOC_LIB)
  set(TBB_LIBRARIES TBB::tbb TBB::tbbmalloc)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB DEFAULT_MSG TBB_INCLUDE_DIRS TBB_LIBRARIES)
