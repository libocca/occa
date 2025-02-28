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
