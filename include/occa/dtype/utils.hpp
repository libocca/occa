#ifndef OCCA_DTYPE_UTILS_HEADER
#define OCCA_DTYPE_UTILS_HEADER

#include <occa/dtype/dtype.hpp>
#include <occa/types/json.hpp>

namespace occa {
  namespace dtype {
    inline json toJson(const dtype_t &dtype,
                       const std::string &name = "") {
      json j;
      dtype.toJson(j, name);
      return j;
    }

    inline dtype_t fromJson(const std::string &str) {
      return dtype_t::fromJson(str);
    }

    inline dtype_t fromJson(const json &j) {
      return dtype_t::fromJson(j);
    }
  }
}

#endif
