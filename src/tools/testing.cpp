#include <cstdio>
#include <cmath>

#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"
#include "occa/tools/testing.hpp"

namespace occa {
  namespace testing {
    template <>
    void compare<float, float>(const float &a, const float &b) {
      const double diff = (a - b)/(fabs(a) + fabs(b) + 1e-50);
      OCCA_ERROR("Comparing Failed",
                 diff < 1e-8);
    }
    template <>
    void compare<double, float>(const double &a, const float &b) {
      const double diff = (a - b)/(fabs(a) + fabs(b) + 1e-50);
      OCCA_ERROR("Comparing Failed",
                 diff < 1e-8);
    }
    template <>
    void compare<float, double>(const float &a, const double &b) {
      const double diff = (a - b)/(fabs(a) + fabs(b) + 1e-50);
      OCCA_ERROR("Comparing Failed",
                 diff < 1e-8);
    }

    template <>
    void compare<double, double>(const double &a, const double &b) {
      const double diff = (a - b)/(fabs(a) + fabs(b) + 1e-50);
      OCCA_ERROR("Comparing Failed",
                 diff < 1e-14);
    }
  }
}
