#include <cstdio>
#include <cmath>

#include "occa/tools/testing.hpp"

namespace occa {
  namespace test {
    template <>
    bool compare<float, float>(const float &a, const float &b) {
      const double diff = (a - b)/(fabs(a) + fabs(b) + 1e-50);
      const bool ret = (diff < 1e-8);
      if (!ret) {
        std::cerr << "a: " << a << '\n'
                  << "b: " << b;
      }
      return ret;
    }
    template <>
    bool compare<double, float>(const double &a, const float &b) {
      const double diff = (a - b)/(fabs(a) + fabs(b) + 1e-50);
      const bool ret = (diff < 1e-8);
      if (!ret) {
        std::cerr << "a: " << a << '\n'
                  << "b: " << b;
      }
      return ret;
    }
    template <>
    bool compare<float, double>(const float &a, const double &b) {
      const double diff = (a - b)/(fabs(a) + fabs(b) + 1e-50);
      const bool ret = (diff < 1e-8);
      if (!ret) {
        std::cerr << "a: " << a << '\n'
                  << "b: " << b;
      }
      return ret;
    }

    template <>
    bool compare<double, double>(const double &a, const double &b) {
      const double diff = (a - b)/(fabs(a) + fabs(b) + 1e-50);
      const bool ret = (diff < 1e-14);
      if (!ret) {
        std::cerr << "a: " << a << '\n'
                  << "b: " << b;
      }
      return ret;
    }
  }
}
