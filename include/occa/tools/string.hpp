/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2016 David Medina and Tim Warburton
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

#ifndef OCCA_TOOLS_STRING_HEADER
#define OCCA_TOOLS_STRING_HEADER

#include <iostream>
#include <iomanip>
#include <sstream>

#include "occa/defines.hpp"
#include "occa/types.hpp"

namespace occa {
  template <class TM>
  inline std::string toString(const TM &t) {
    std::stringstream ss;
    ss << t;
    return ss.str();
  }

  template <>
  inline std::string toString<std::string>(const std::string &t) {
    return t;
  }

  template <>
  inline std::string toString<float>(const float &t) {
    std::stringstream ss;
    ss << std::scientific << std::setprecision(8) << t << 'f';
    return ss.str();
  }

  template <>
  inline std::string toString<double>(const double &t) {
    std::stringstream ss;
    ss << std::scientific << std::setprecision(16) << t;
    return ss.str();
  }

  template <class TM>
  inline TM fromString(const std::string &s) {
    std::stringstream ss;
    TM t;
    ss << s;
    ss >> t;
    return t;
  }

  template <>
  inline std::string fromString(const std::string &s) {
    return s;
  }

  udim_t atoi(const char *c);
  udim_t atoiBase2(const char *c);
  udim_t atoi(const std::string &str);

  double atof(const char *c);
  double atof(const std::string &str);

  double atod(const char *c);
  double atod(const std::string &str);

  std::string stringifyBytes(udim_t bytes);

  strVector_t split(const std::string &s, const char delimeter);
}

#endif
