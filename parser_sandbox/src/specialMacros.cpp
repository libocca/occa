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
  namespace lang {
    // defined()
    definedMacro::definedMacro(preprocessor &pp_) :
      macro_t(pp_, "defined") {}

    bool definedMacro::expandTokens(identifierToken &source,
                                    tokenVector &expandedTokens) {
      bool isDefined = !!pp.getMacro(source.value);
      expandedTokens.push_back(
        new primitiveToken(source.origin,
                           isDefined,
                           isDefined ? "true" : "false")
      );
      return true;
    }

    // __FILE__
    fileMacro::fileMacro(preprocessor &pp_) :
      macro_t(pp_, "__FILE__") {}

    bool fileMacro::expandTokens(identifierToken &source,
                                 tokenVector &expandedTokens) {
      expandedTokens.push_back(
        new stringToken(source.origin,
                        source.origin.file->filename)
      );
      return true;
    }

    // __LINE__
    lineMacro::lineMacro(preprocessor &pp_) :
      macro_t(pp_, "__LINE__") {}

    bool lineMacro::expandTokens(identifierToken &source,
                                 tokenVector &expandedTokens) {
      const int line = source.origin.position.line;
      expandedTokens.push_back(
        new primitiveToken(source.origin,
                           line,
                           occa::toString(line))
      );
      return true;
    }

    // __DATE__
    dateMacro::dateMacro(preprocessor &pp_) :
      macro_t(pp_, "__DATE__") {}

    bool dateMacro::expandTokens(identifierToken &source,
                                 tokenVector &expandedTokens) {
      static char month[12][5] = {
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
      };

      time_t t = ::time(NULL);
      struct tm *ct = ::localtime(&t);

      std::stringstream ss;

      if (ct == NULL) {
        ss << "??? ?? ????";
      } else {
        ss << month[ct->tm_mon] << ' ';
        if (ct->tm_mday < 10) {
          ss << ' ';
        }
        ss << ct->tm_mday << ' '
           << ct->tm_year + 1900;
      }

      expandedTokens.push_back(
        new stringToken(source.origin,
                        ss.str())
      );
      return true;
    }

    // __TIME__
    timeMacro::timeMacro(preprocessor &pp_) :
      macro_t(pp_, "__TIME__") {}

    bool timeMacro::expandTokens(identifierToken &source,
                                 tokenVector &expandedTokens) {
      time_t t = ::time(NULL);
      struct tm *ct = ::localtime(&t);

      std::stringstream ss;
      if (ct == NULL) {
        ss << "??:??:??";
      } else {
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
      }

      expandedTokens.push_back(
        new stringToken(source.origin,
                        ss.str())
      );
      return true;
    }

    // __COUNTER__
    counterMacro::counterMacro(preprocessor &pp_) :
      macro_t(pp_, "__COUNTER__"),
      counter(0) {}

    bool counterMacro::expandTokens(identifierToken &source,
                                    tokenVector &expandedTokens) {
      const int value = counter;
      expandedTokens.push_back(
        new primitiveToken(source.origin,
                           value,
                           occa::toString(value))
      );
      return true;
    }
  }
}
