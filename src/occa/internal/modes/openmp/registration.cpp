#include <occa/defines.hpp>

#if OCCA_OPENMP_ENABLED
#  include <omp.h>
#endif

#include <occa/internal/modes/openmp/registration.hpp>

namespace occa {
  namespace openmp {
    openmpMode::openmpMode() :
        mode_t("OpenMP") {}

    bool openmpMode::init() {
#if OCCA_OPENMP_ENABLED
      // Generate an OpenMP library dependency (so it doesn't crash when dlclose())
      omp_get_num_threads();
      return true;
#else
      return false;
#endif
    }

    modeDevice_t* openmpMode::newDevice(const occa::json &props) {
      return new device(setModeProp(props));
    }

    int openmpMode::getDeviceCount(const occa::json &props) {
      return 1;
    }

    openmpMode mode;
  }
}
