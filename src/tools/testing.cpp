#include <cstdio>
#include <cmath>

#include <occa/tools/testing.hpp>

namespace occa {
  namespace test {
    template <>
    bool areEqual<float, float>(const float &a, const float &b) {
      const double diff = (a - b)/(fabs(a) + fabs(b) + 1e-50);
      return (fabs(diff) < 1e-8);
    }
    template <>
    bool areEqual<double, float>(const double &a, const float &b) {
      const double diff = (a - b)/(fabs(a) + fabs(b) + 1e-50);
      return (fabs(diff) < 1e-8);
    }
    template <>
    bool areEqual<float, double>(const float &a, const double &b) {
      const double diff = (a - b)/(fabs(a) + fabs(b) + 1e-50);
      return (fabs(diff) < 1e-8);
    }

    template <>
    bool areEqual<double, double>(const double &a, const double &b) {
      const double diff = (a - b)/(fabs(a) + fabs(b) + 1e-50);
      return (fabs(diff) < 1e-14);
    }

    template <>
    bool areEqual<const char*, const char*>(const char * const &a,
                                            const char * const &b) {
      return (std::string(a) == std::string(b));
    }
  }
}
