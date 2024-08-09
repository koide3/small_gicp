#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "small_gicp::small_gicp" for configuration "Release"
set_property(TARGET small_gicp::small_gicp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(small_gicp::small_gicp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsmall_gicp.so"
  IMPORTED_SONAME_RELEASE "libsmall_gicp.so"
  )

list(APPEND _cmake_import_check_targets small_gicp::small_gicp )
list(APPEND _cmake_import_check_files_for_small_gicp::small_gicp "${_IMPORT_PREFIX}/lib/libsmall_gicp.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
