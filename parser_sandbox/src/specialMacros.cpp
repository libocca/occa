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
#include <sstream>
#include <ctime>

#include "occa/tools/string.hpp"

#include "preprocessor.hpp"
#include "specialMacros.hpp"

namespace occa {
  // __FILE__
  fileMacro::fileMacro(const preprocessor_t *preprocessor_) :
    macro_t(preprocessor_) {
    name = "__FILE__";
  }

  std::string fileMacro::expand(char *&c) const {
    return preprocessor->currentFrame.filename();
  }

  // __LINE__
  lineMacro::lineMacro(const preprocessor_t *preprocessor_) :
    macro_t(preprocessor_) {
    name = "__LINE__";
  }

  std::string lineMacro::expand(char *&c) const {
    return occa::toString(preprocessor->currentFrame.lineNumber);
  }

  // __DATE__
  dateMacro::dateMacro(const preprocessor_t *preprocessor_) :
    macro_t(preprocessor_) {
    name = "__DATE__";
  }

  std::string dateMacro::expand(char *&c) const {
    static char month[12][5] = {
      "Jan", "Feb", "Mar", "Apr", "May", "Jun",
      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    };

    time_t t = ::time(NULL);
    struct tm *ct = ::localtime(&t);

    if (ct == NULL) {
      return "??? ?? ????";
    }

    std::stringstream ss;
    ss << month[ct->tm_mon] << ' ';
    if (ct->tm_mday < 10) {
      ss << ' ';
    }
    ss << ct->tm_mday << ' '
       << ct->tm_year + 1900;
    return ss.str();
  }

  // __TIME__
  timeMacro::timeMacro(const preprocessor_t *preprocessor_) :
    macro_t(preprocessor_) {
    name = "__TIME__";
  }

  std::string timeMacro::expand(char *&c) const {
    time_t t = ::time(NULL);
    struct tm *ct = ::localtime(&t);

    if (ct == NULL) {
      return "??:??:??";
    }

    std::stringstream ss;
    if (ct->tm_hour < 10) {
      ss << '0';
    }
    ss << ct->tm_hour << ':';
    if (ct->tm_min < 10) {
      ss << '0';
    }
    ss << ct->tm_min << ':';
    if (ct->tm_sec < 10) {
      ss << '0';
    }
    ss << ct->tm_sec;
    return ss.str();
  }

  // __COUNTER__
  counterMacro::counterMacro(const preprocessor_t *preprocessor_) :
    macro_t(preprocessor_),
    counter(0) {
    name = "__COUNTER__";
  }

  std::string counterMacro::expand(char *&c) const {
    return occa::toString(counter++);
  }
}
