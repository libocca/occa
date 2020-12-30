#ifndef OCCA_UTILS_ENV_HEADER
#define OCCA_UTILS_ENV_HEADER

#include <occa/types/properties.hpp>

namespace occa {
  properties& settings();

  namespace env {
    extern std::string OCCA_DIR, OCCA_INSTALL_DIR, OCCA_CACHE_DIR;

    void setOccaCacheDir(const std::string &path);
  }
}

#endif
