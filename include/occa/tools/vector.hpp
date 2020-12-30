#ifndef OCCA_INTERNAL_UTILS_VECTOR_HEADER
#define OCCA_INTERNAL_UTILS_VECTOR_HEADER

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

  template <>
  dim_t indexOf(const std::vector<float> &vec, const float &value);

  template <>
  dim_t indexOf(const std::vector<double> &vec, const double &value);

  template <>
  dim_t indexOf(const std::vector<long double> &vec, const long double &value);
}

#endif
