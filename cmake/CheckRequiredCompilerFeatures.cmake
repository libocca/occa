include(CheckCXXCompilerFlag)

check_cxx_source_compiles("
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
  return 0;
}
" HAS_FILE_SYSTEM)

if (NOT HAS_FILE_SYSTEM)
  message(FATAL_ERROR "CXX compiler doesn't have std::filesystem support !")
endif()

if (OCCA_DPCPP_ENABLED)
  check_cxx_compiler_flag("-fsycl" IS_SYCL_SUPPORTED)
  if (NOT IS_SYCL_SUPPORTED)
    message(FATAL_ERROR "DPCPP is enabled but the compiler doesn't support \"-fsycl\" flag!")
  endif()
endif()
