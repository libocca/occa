/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
#include <algorithm>

#include <occa/defines.hpp>
#include <occa/tools/string.hpp>
#include <occa/tools/exception.hpp>
#include <occa/tools/sys.hpp>

#define ASSERT_LE(a, b)                                 \
  OCCA_ERROR("Assertion Failed: Value is >",            \
             occa::test::assertLessThanOrEqual(a, b));

#define ASSERT_LT(a, b)                         \
  OCCA_ERROR("Assertion Failed: Value is >=",   \
             occa::test::assertLessThan(a, b));

#define ASSERT_GT(a, b)                             \
  OCCA_ERROR("Assertion Failed: Value is <=",       \
             occa::test::assertGreaterThan(a, b));

#define ASSERT_GE(a, b)                                   \
  OCCA_ERROR("Assertion Failed: Value is <",              \
             occa::test::assertGreaterThanOrEqual(a, b));

#define ASSERT_EQ(a, b)                                 \
  OCCA_ERROR("Assertion Failed: Values are not equal",  \
             occa::test::assertEqual(a, b));

#define ASSERT_NEQ(a, b)                            \
  OCCA_ERROR("Assertion Failed: Values are equal",  \
             occa::test::assertNotEqual(a, b));

#define ASSERT_EQ_BINARY(a, b)                          \
  OCCA_ERROR("Assertion Failed: Values are not equal",  \
             occa::test::assertEqualBinary(a, b));

#define ASSERT_NEQ_BINARY(a, b)                       \
  OCCA_ERROR("Assertion Failed: Values are equal",    \
             occa::test::assertNotEqualBinary(a, b));

#define ASSERT_TRUE(value)                        \
  OCCA_ERROR("Assertion Failed: Value not true",  \
             (bool) (value));

#define ASSERT_FALSE(value)                       \
  OCCA_ERROR("Assertion Failed: Value not false", \
             !((bool) (value)));

#define ASSERT_IN(value, vec)                                       \
  OCCA_ERROR("Assertion Failed: Value not in",                      \
             std::find(vec.begin(), vec.end(), value) != vec.end())

#define ASSERT_NOT_IN(value, vec)                                   \
  OCCA_ERROR("Assertion Failed: Value in",                          \
             std::find(vec.begin(), vec.end(), value) == vec.end())

#define ASSERT_THROW_START                      \
  do {                                          \
    bool threw = false;                         \
    try

#define ASSERT_THROW_END                                      \
    catch (occa::exception exc) {                             \
      threw = true;                                           \
    }                                                         \
    OCCA_ERROR("Assertion Failed: No occa::exception thrown", \
               threw);                                        \
  } while(0)

#define ASSERT_THROW(source)                    \
  ASSERT_THROW_START {                          \
    source                                      \
  } ASSERT_THROW_END

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
    bool areEqual<double, double>(const double &a, const double &b);

    template <>
    bool areEqual<const char*, const char*>(const char * const &a,
                                            const char * const &b);
  }
}
