#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"

#define OCCA_TEST_COMPARE(a, b) OCCA_ERROR("Comparison Failed", \
                                           occa::test::compare(a, b));

namespace occa {
  namespace test {
    template <class TM1, class TM2>
    bool compare(const TM1 &a, const TM2 &b) {
      if (a != b) {
        std::cerr << "a: " << a << '\n'
                  << "b: " << b;
      }
      return (a == b);
    }

    template <>
    bool compare<float, float>(const float &a, const float &b);

    template <>
    bool compare<double, float>(const double &a, const float &b);

    template <>
    bool compare<float, double>(const float &a, const double &b);

    template <>
    bool compare<double, double>(const double &a, const double &b);
  }
}
