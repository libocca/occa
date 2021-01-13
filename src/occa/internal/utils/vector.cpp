#include <occa/internal/utils/string.hpp>
#include <occa/internal/utils/vector.hpp>
#include <occa/types/bits.hpp>

namespace occa {
  template <>
  dim_t indexOf(const std::vector<float> &vec, const float &value) {
    const dim_t size = vec.size();
    for (dim_t i = 0; i < size; ++i) {
      if (areBitwiseEqual(vec[i], value)) {
        return i;
      }
    }
    return -1;
  }

  template <>
  dim_t indexOf(const std::vector<double> &vec, const double &value) {
    const dim_t size = vec.size();
    for (dim_t i = 0; i < size; ++i) {
      if (areBitwiseEqual(vec[i], value)) {
        return i;
      }
    }
    return -1;
  }

  template <>
  dim_t indexOf(const std::vector<long double> &vec, const long double &value) {
    const dim_t size = vec.size();
    for (dim_t i = 0; i < size; ++i) {
      if (areBitwiseEqual(vec[i], value)) {
        return i;
      }
    }
    return -1;
  }
}
