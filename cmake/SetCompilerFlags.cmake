include(CheckCXXCompilerFlag)

function(set_optional_cxx_flag var)
  foreach(flag ${ARGN})
    string(MAKE_C_IDENTIFIER "Allowed_CXX_Flag${flag}" check_var)
    check_cxx_compiler_flag("${flag}" ${check_var})
    if (${${check_var}})
      set(${var} "${${var}} ${flag}")
    endif()
  endforeach()
  set(${var} "${${var}}" PARENT_SCOPE)
endfunction(set_optional_cxx_flag)

# Enable warnings
set_optional_cxx_flag(SUPPORTED_WARN_CXX_FLAGS "-Wall -Wextra")
set_optional_cxx_flag(SUPPORTED_WARN_CXX_FLAGS "-Wunused-function -Wunused-variable")
set_optional_cxx_flag(SUPPORTED_WARN_CXX_FLAGS "-Wwrite-strings -Wfloat-equal" "-Wcast-align -Wlogical-op" "-Wshadow")
# set_optional_cxx_flag(SUPPORTED_WARN_CXX_FLAGS "-Wstrict-prototypes -Wmissing-prototypes" "-Wundef")
# Disable warnings
set_optional_cxx_flag(SUPPORTED_WARN_CXX_FLAGS "-Wno-unused-parameter")
set_optional_cxx_flag(SUPPORTED_WARN_CXX_FLAGS "-diag-disable 11074 -diag-disable 11076")   # Intel: Disable warnings about inline limits reached

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SUPPORTED_WARN_CXX_FLAGS}")

set_optional_cxx_flag(SUPPORTED_WERROR_CXX_FLAGS "-Werror")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${SUPPORTED_WERROR_CXX_FLAGS}")

include(CheckCCompilerFlag)

function(set_optional_c_flag var)
  foreach(flag ${ARGN})
    string(MAKE_C_IDENTIFIER "Allowed_C_Flag${flag}" check_var)
    check_c_compiler_flag("${flag}" ${check_var})
    if (${${check_var}})
      set(${var} "${${var}} ${flag}")
    endif()
  endforeach()
  set(${var} "${${var}}" PARENT_SCOPE)
endfunction(set_optional_c_flag)

# Enable warnings
set_optional_c_flag(SUPPORTED_WARN_C_FLAGS "-Wall -Wextra")
set_optional_c_flag(SUPPORTED_WARN_C_FLAGS "-Wunused-function -Wunused-variable")
set_optional_c_flag(SUPPORTED_WARN_C_FLAGS "-Wwrite-strings -Wfloat-equal" "-Wcast-align -Wlogical-op" "-Wshadow")
# set_optional_c_flag(SUPPORTED_WARN_C_FLAGS "-Wstrict-prototypes -Wmissing-prototypes" "-Wundef")
# Disble warnings
set_optional_c_flag(SUPPORTED_WARN_C_FLAGS "-Wno-c++11-long-long")
set_optional_c_flag(SUPPORTED_WARN_C_FLAGS "-diag-disable 11074 -diag-disable 11076")   # Disable warnings about inline limits reached

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SUPPORTED_WARN_C_FLAGS}")

set_optional_c_flag(SUPPORTED_WERROR_C_FLAGS "-Werror")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${SUPPORTED_WERROR_C_FLAGS}")
