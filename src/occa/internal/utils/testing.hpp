#include <algorithm>

#include <occa/defines.hpp>
#include <occa/types/bits.hpp>
#include <occa/internal/utils/string.hpp>
#include <occa/utils/exception.hpp>
#include <occa/internal/utils/sys.hpp>

#define OCCA_ASSERT_ERROR(message, expr)                \
  do {                                                  \
    bool exprThrewException = true;                     \
    try {                                               \
      const bool assertIsOk = (bool) (expr);            \
      exprThrewException = false;                       \
      OCCA_ERROR(message, assertIsOk);                  \
    } catch (occa::exception& exc) {                    \
      if (exprThrewException) {                         \
        /* Print expr exception and assert exception*/  \
        std::cerr << exc << '\n';                       \
        OCCA_FORCE_ERROR(message);                      \
      } else {                                          \
        /* Re-throw exception */                        \
        throw exc;                                      \
      }                                                 \
    }                                                   \
  } while (false)

#define ASSERT_LE(a, b)                                       \
  OCCA_ASSERT_ERROR("Assertion Failed: Value is >",           \
                    occa::test::assertLessThanOrEqual(a, b))

#define ASSERT_LT(a, b)                               \
  OCCA_ASSERT_ERROR("Assertion Failed: Value is >=",  \
                    occa::test::assertLessThan(a, b))

#define ASSERT_GT(a, b)                                   \
  OCCA_ASSERT_ERROR("Assertion Failed: Value is <=",      \
                    occa::test::assertGreaterThan(a, b))

#define ASSERT_GE(a, b)                                         \
  OCCA_ASSERT_ERROR("Assertion Failed: Value is <",             \
                    occa::test::assertGreaterThanOrEqual(a, b))

#define ASSERT_EQ(a, b)                                       \
  OCCA_ASSERT_ERROR("Assertion Failed: Values are not equal", \
                    occa::test::assertEqual(a, b))

#define ASSERT_NEQ(a, b)                                  \
  OCCA_ASSERT_ERROR("Assertion Failed: Values are equal", \
                    occa::test::assertNotEqual(a, b))

#define ASSERT_EQ_BINARY(a, b)                                \
  OCCA_ASSERT_ERROR("Assertion Failed: Values are not equal", \
                    occa::test::assertEqualBinary(a, b))

#define ASSERT_NEQ_BINARY(a, b)                             \
  OCCA_ASSERT_ERROR("Assertion Failed: Values are equal",   \
                    occa::test::assertNotEqualBinary(a, b))

#define ASSERT_TRUE(value)                              \
  OCCA_ASSERT_ERROR("Assertion Failed: Value not true", \
                    (bool) (value))

#define ASSERT_FALSE(value)                               \
  OCCA_ASSERT_ERROR("Assertion Failed: Value not false",  \
                    !((bool) (value)))

#define ASSERT_IN(value, vec)                                           \
  OCCA_ASSERT_ERROR("Assertion Failed: Value not in",                   \
                    std::find(vec.begin(), vec.end(), value) != vec.end())

#define ASSERT_NOT_IN(value, vec)                                       \
  OCCA_ASSERT_ERROR("Assertion Failed: Value in",                       \
                    std::find(vec.begin(), vec.end(), value) == vec.end())

#define ASSERT_THROW(source)                                  \
  do {                                                        \
    bool threw = false;                                       \
    try {                                                     \
      source;                                                 \
    } catch (occa::exception&) {                              \
      threw = true;                                           \
    }                                                         \
    OCCA_ERROR("Assertion Failed: No occa::exception thrown", \
               threw);                                        \
  } while (false)

namespace occa {
  namespace test {
    template <class TM1, class TM2>
    bool areEqual(const TM1 &a, const TM2 &b) {
      return (a == b);
    }

    template <class TM1, class TM2>
    bool assertLessThanOrEqual(const TM1 &a, const TM2 &b) {
      if ((a < b) || areEqual(a, b)) {
        return true;
      }
      std::cerr << "left : [" << a << "]\n"
                << "right: [" << b << "]\n"
                << std::flush;
      return false;
    }

    template <class TM1, class TM2>
    bool assertLessThan(const TM1 &a, const TM2 &b) {
      if (a < b) {
        return true;
      }
      std::cerr << "left : [" << a << "]\n"
                << "right: [" << b << "]\n"
                << std::flush;
      return false;
    }

    template <class TM1, class TM2>
    bool assertGreaterThan(const TM1 &a, const TM2 &b) {
      if (!((a < b) || areEqual(a, b))) {
        return true;
      }
      std::cerr << "left : [" << a << "]\n"
                << "right: [" << b << "]\n"
                << std::flush;
      return false;
    }

    template <class TM1, class TM2>
    bool assertGreaterThanOrEqual(const TM1 &a, const TM2 &b) {
      if (!(a < b)) {
        return true;
      }
      std::cerr << "left : [" << a << "]\n"
                << "right: [" << b << "]\n"
                << std::flush;
      return false;
    }

    template <class TM1, class TM2>
    bool assertEqual(const TM1 &a, const TM2 &b) {
      if (areEqual(a, b)) {
        return true;
      }
      std::cerr << "left : [" << a << "]\n"
                << "right: [" << b << "]\n"
                << std::flush;
      return false;
    }

    template <class TM1, class TM2>
    bool assertNotEqual(const TM1 &a, const TM2 &b) {
      if (!areEqual(a, b)) {
        return true;
      }
      std::cerr << "left : [" << a << "]\n"
                << "right: [" << b << "]\n"
                << std::flush;
      return false;
    }

    template <class TM1, class TM2>
    bool assertEqualBinary(const TM1 &a, const TM2 &b) {
      if (a == b) {
        return true;
      }
      std::cerr << "left : [" << stringifySetBits(a) << "]\n"
                << "right: [" << stringifySetBits(b) << "]\n"
                << std::flush;
      return false;
    }

    template <class TM1, class TM2>
    bool assertNotEqualBinary(const TM1 &a, const TM2 &b) {
      if (a != b) {
        return true;
      }
      std::cerr << "left : [" << stringifySetBits(a) << "]\n"
                << "right: [" << stringifySetBits(b) << "]\n"
                << std::flush;
      return false;
    }

    template <>
    bool areEqual<float, float>(const float &a, const float &b);

    template <>
    bool areEqual<double, float>(const double &a, const float &b);

    template <>
    bool areEqual<float, double>(const float &a, const double &b);

    template <>
    bool areEqual<long double, float>(const long double &a, const float &b);

    template <>
    bool areEqual<float, long double>(const float &a, const long double &b);

    template <>
    bool areEqual<double, double>(const double &a, const double &b);

    template <>
    bool areEqual<long double, double>(const long double &a, const double &b);

    template <>
    bool areEqual<double, long double>(const double &a, const long double &b);

    template <>
    bool areEqual<long double, long double>(const long double &a, const long double &b);

    template <>
    bool areEqual<const char*, const char*>(const char * const &a,
                                            const char * const &b);
  }
}
