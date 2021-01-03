#ifndef OCCA_UTILS_IO_HEADER
#define OCCA_UTILS_IO_HEADER

#include <iostream>

namespace occa {
  namespace io {
    bool exists(const std::string &filename);

    void addLibraryPath(const std::string &library,
                        const std::string &path);
  }
}

#endif
