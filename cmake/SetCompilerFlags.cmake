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
set_optional_cxx_flag(SUPPORTED_WARN_CXX_FLAGS "--display_error_number")  # Show PGI error numbers

if (CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
  # Workaround for
  # CMakeFiles/tools-trie.dir/trie.cpp.o: in function `occa::trie<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >::getLongest(char const*, int) const':
  # occa/include/occa/tools/trie.tpp:238:(.text+0x5cdc): relocation truncated to fit: R_X86_64_PC32 against `.rodata'
  set(CMAKE_CXX_FLAGS_RELEASE "-O1 -DNDEBUG")
endif()

set(CMAKE_CXX_FLAGS "${SUPPORTED_WARN_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

set_optional_cxx_flag(SUPPORTED_WERROR_CXX_FLAGS "-Werror")
set(CMAKE_CXX_FLAGS_DEBUG "${SUPPORTED_WERROR_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")

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
set_optional_c_flag(SUPPORTED_WARN_C_FLAGS "--display_error_number")  # Show PGI error numbers

set(CMAKE_C_FLAGS "${SUPPORTED_WARN_C_FLAGS} ${CMAKE_C_FLAGS}")

set_optional_c_flag(SUPPORTED_WERROR_C_FLAGS "-Werror")
set(CMAKE_C_FLAGS_DEBUG "${SUPPORTED_WERROR_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")

if (ENABLE_FORTRAN)
  include(CheckFortranCompilerFlag)

  function(set_optional_fortran_flag var)
    foreach(flag ${ARGN})
      string(MAKE_C_IDENTIFIER "Allowed_Fortran_Flag${flag}" check_var)
      check_fortran_compiler_flag("${flag}" ${check_var})
      if (${${check_var}})
        set(${var} "${${var}} ${flag}")
      endif()
    endforeach()
    set(${var} "${${var}}" PARENT_SCOPE)
  endfunction(set_optional_fortran_flag)

  # Enable warnings
  set_optional_fortran_flag(SUPPORTED_WARN_Fortran_FLAGS "-Wall -Wextra")
  set_optional_fortran_flag(SUPPORTED_WARN_Fortran_FLAGS "-warn all")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${SUPPORTED_WARN_Fortran_FLAGS}")

  set_optional_fortran_flag(SUPPORTED_WERROR_Fortran_FLAGS "-Werror")
  set_optional_fortran_flag(SUPPORTED_WERROR_Fortran_FLAGS "-warn errors")
  set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} ${SUPPORTED_WERROR_Fortran_FLAGS}")

  set_optional_fortran_flag(SUPPORTED_WNO_INTEGER_DIVISION_Fortran_FLAGS "-Wno-integer-division")

endif()
