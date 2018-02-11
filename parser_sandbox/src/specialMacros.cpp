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

    tokenMap& definedMacro::cloneMap() const {
      return *(new definedMacro(pp));
    }

    token_t* definedMacro::pop() {
      fileOrigin origin; // TODO

      token_t *token = pp.getSourceToken();
      if (!(token_t::safeType(token) & tokenType::identifier)) {
        return token;
      }

      bool isDefined = !!pp.getMacro(token->to<identifierToken>().value);
      return new primitiveToken(origin,
                                isDefined,
                                isDefined ? "true" : "false");
    }

    // __FILE__
    fileMacro::fileMacro(preprocessor &pp_) :
      macro_t(pp_, "__FILE__") {}

    tokenMap& fileMacro::cloneMap() const {
      return *(new fileMacro(pp));
    }

    token_t* fileMacro::pop() {
      fileOrigin origin; // TODO
      return new stringToken(origin, "file");
    }

    // __LINE__
    lineMacro::lineMacro(preprocessor &pp_) :
      macro_t(pp_, "__LINE__") {}

    tokenMap& lineMacro::cloneMap() const {
      return *(new lineMacro(pp));
    }

    token_t* lineMacro::pop() {
      fileOrigin origin; // TODO
      const primitive value = 0;
      const std::string strValue = occa::toString(value);
      return new primitiveToken(origin, value, strValue);
    }

    // __DATE__
    dateMacro::dateMacro(preprocessor &pp_) :
      macro_t(pp_, "__DATE__") {}

    tokenMap& dateMacro::cloneMap() const {
      return *(new dateMacro(pp));
    }

    token_t* dateMacro::pop() {
      static char month[12][5] = {
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
      };

      fileOrigin origin; // TODO

      time_t t = ::time(NULL);
      struct tm *ct = ::localtime(&t);

      if (ct == NULL) {
        return new stringToken(origin, "??? ?? ????");
      }

      std::stringstream ss;
      ss << month[ct->tm_mon] << ' ';
      if (ct->tm_mday < 10) {
        ss << ' ';
      }
      ss << ct->tm_mday << ' '
         << ct->tm_year + 1900;

      return new stringToken(origin, ss.str());
    }

    // __TIME__
    timeMacro::timeMacro(preprocessor &pp_) :
      macro_t(pp_, "__TIME__") {}

    tokenMap& timeMacro::cloneMap() const {
      return *(new timeMacro(pp));
    }

    token_t* timeMacro::pop() {
      fileOrigin origin; // TODO

      time_t t = ::time(NULL);
      struct tm *ct = ::localtime(&t);

      if (ct == NULL) {
        return new stringToken(origin, "??:??:??");
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

      return new stringToken(origin, ss.str());
    }

    // __COUNTER__
    counterMacro::counterMacro(preprocessor &pp_) :
      macro_t(pp_, "__COUNTER__"),
      counter(0) {}

    tokenMap& counterMacro::cloneMap() const {
      return *(new counterMacro(pp));
    }

    token_t* counterMacro::pop() {
      fileOrigin origin; // TODO
      const primitive value = counter++;
      const std::string strValue = occa::toString(value);
      return new primitiveToken(origin, value, strValue);
    }
  }
}
