#ifndef OCCA_TOOLS_VECTOR_HEADER
#define OCCA_TOOLS_VECTOR_HEADER

#include <string>
#include <vector>

#include <occa/types/typedefs.hpp>

namespace occa {
  template <class TM>
  dim_t indexOf(const std::vector<TM> &vec,
              const TM &value) {
    const dim_t size = vec.size();
    for (dim_t i = 0; i < size; ++i) {
      if (vec[i] == value) {
        return i;
      }
    }
    return -1;
  }

  std::string join(const std::vector<std::string> &vec,
                   const std::string &delimiter);
}

#endif
