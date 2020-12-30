#include <cstdio>
#include <cmath>

#include <occa/internal/utils/testing.hpp>

namespace occa {
  namespace test {
    template <>
    bool areEqual<float, float>(const float &a, const float &b) {
      const double diff = (a - b)/(std::abs(a) + std::abs(b) + 1e-50);
      return (std::abs(diff) < 1e-8);
    }
    template <>
    bool areEqual<double, float>(const double &a, const float &b) {
      const double diff = (a - b)/(std::abs(a) + std::abs(b) + 1e-50);
      return (std::abs(diff) < 1e-8);
    }
    template <>
    bool areEqual<float, double>(const float &a, const double &b) {
      const double diff = (a - b)/(std::abs(a) + std::abs(b) + 1e-50);
      return (std::abs(diff) < 1e-8);
    }

    template <>
    bool areEqual<long double, float>(const long double &a, const float &b) {
      const double diff = (a - b)/(std::abs(a) + std::abs(b) + 1e-50);
      return (std::abs(diff) < 1e-8);
    }

    template <>
    bool areEqual<float, long double>(const float &a, const long double &b) {
      const double diff = (a - b)/(std::abs(a) + std::abs(b) + 1e-50);
      return (std::abs(diff) < 1e-8);
    }

    template <>
    bool areEqual<double, double>(const double &a, const double &b) {
      const double diff = (a - b)/(std::abs(a) + std::abs(b) + 1e-50);
      return (std::abs(diff) < 1e-14);
    }

    template <>
    bool areEqual<long double, double>(const long double &a, const double &b) {
      const double diff = (a - b)/(std::abs(a) + std::abs(b) + 1e-50);
      return (std::abs(diff) < 1e-14);
    }

    template <>
    bool areEqual<double, long double>(const double &a, const long double &b) {
      const double diff = (a - b)/(std::abs(a) + std::abs(b) + 1e-50);
      return (std::abs(diff) < 1e-14);
    }

    template <>
    bool areEqual<long double, long double>(const long double &a, const long double &b) {
      const double diff = (a - b)/(std::abs(a) + std::abs(b) + 1e-50);
      return (std::abs(diff) < 1e-14);
    }

    template <>
    bool areEqual<const char*, const char*>(const char * const &a,
                                            const char * const &b) {
      return (std::string(a) == std::string(b));
    }
  }
}
