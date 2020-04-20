#include <occa/tools/string.hpp>
#include <occa/tools/vector.hpp>
#include <occa/types/fp.hpp>

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

  std::string join(const std::vector<std::string> &vec,
                   const std::string &delimiter) {
    const size_t size = vec.size();
    std::string ret;
    for (size_t i = 0; i < size; ++i) {
      if (i) {
        ret += delimiter;
      }
      ret += toString(vec[i]);
    }
    return ret;
  }
}
