#include <occa/defines.hpp>

#if OCCA_OPENMP_ENABLED
#  ifndef OCCA_MODES_OPENMP_UTILS_HEADER
#  define OCCA_MODES_OPENMP_UTILS_HEADER

namespace occa {
  namespace openmp {
    extern std::string notSupported;

    std::string baseCompilerFlag(const int vendor_);
    std::string compilerFlag(const int vendor_,
                             const std::string &compiler);
  }
}

#  endif
#endif
