#include <occa/tools/string.hpp>
#include <occa/tools/vector.hpp>

namespace occa {
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
