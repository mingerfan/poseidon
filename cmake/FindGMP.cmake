include(FindPackageHandleStandardArgs)

set(GMP_ROOT "" CACHE PATH "GMP installation prefix")

set(_GMP_HINTS)
if(GMP_ROOT)
    list(APPEND _GMP_HINTS "${GMP_ROOT}")
endif()
if(DEFINED ENV{GMP_ROOT} AND NOT "$ENV{GMP_ROOT}" STREQUAL "")
    list(APPEND _GMP_HINTS "$ENV{GMP_ROOT}")
endif()

find_path(GMP_INCLUDE_DIR
    NAMES gmp.h
    HINTS ${_GMP_HINTS}
    PATH_SUFFIXES include include/lib)
find_path(GMPXX_INCLUDE_DIR
    NAMES gmpxx.h
    HINTS ${_GMP_HINTS}
    PATH_SUFFIXES include include/lib)
find_library(GMP_LIBRARY
    NAMES gmp
    HINTS ${_GMP_HINTS}
    PATH_SUFFIXES lib lib64)
find_library(GMPXX_LIBRARY
    NAMES gmpxx
    HINTS ${_GMP_HINTS}
    PATH_SUFFIXES lib lib64)

find_package_handle_standard_args(GMP
    "Could NOT find the GMP C++ development files. Install GMP with C++ support or set GMP_ROOT to its installation prefix."
    GMP_INCLUDE_DIR GMPXX_INCLUDE_DIR GMP_LIBRARY GMPXX_LIBRARY)

if(GMP_FOUND)
    set(GMP_INCLUDE_DIRS "${GMP_INCLUDE_DIR};${GMPXX_INCLUDE_DIR}")
    list(REMOVE_DUPLICATES GMP_INCLUDE_DIRS)
    set(GMP_LIBRARIES "${GMPXX_LIBRARY};${GMP_LIBRARY}")

    if(NOT TARGET GMP::gmp)
        add_library(GMP::gmp UNKNOWN IMPORTED)
        set_target_properties(GMP::gmp PROPERTIES
            IMPORTED_LOCATION "${GMP_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIR}")
    endif()

    if(NOT TARGET GMP::gmpxx)
        add_library(GMP::gmpxx UNKNOWN IMPORTED)
        set_target_properties(GMP::gmpxx PROPERTIES
            IMPORTED_LOCATION "${GMPXX_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${GMP_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES GMP::gmp)
    endif()
endif()

mark_as_advanced(GMP_INCLUDE_DIR GMPXX_INCLUDE_DIR GMP_LIBRARY GMPXX_LIBRARY)
