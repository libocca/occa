#include <occa/defines.hpp>

#ifndef OCCA_INTERNAL_MODES_OPENMP_UTILS_HEADER
#define OCCA_INTERNAL_MODES_OPENMP_UTILS_HEADER

namespace occa {
  namespace openmp {
    extern std::string notSupported;

    std::string baseCompilerFlag(const int vendor_);
    std::string compilerFlag(const int vendor_,
                             const std::string &compiler);
  }
}

#endif
