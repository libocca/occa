#include <occa/defines.hpp>

#if OCCA_OPENMP_ENABLED

#include <omp.h>

#include <occa/mode/openmp/registration.hpp>

namespace occa {
  namespace openmp {
    modeInfo::modeInfo() {}

    bool modeInfo::init() {
      // Generate an OpenMP library dependency (so it doesn't crash when dlclose())
      omp_get_num_threads();
      return true;
    }

    occa::mode<openmp::modeInfo,
               openmp::device> mode("OpenMP");
  }
}

#endif
