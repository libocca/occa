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
#include "occa/tools/io.hpp"

#include "preprocessor.hpp"
#include "specialMacros.hpp"

namespace occa {
  namespace lang {
    // defined()
    definedMacro::definedMacro(preprocessor &pp_) :
      macro_t(pp_, "defined") {
      argNames.add("MACRO_NAME", 0);
    }

    void definedMacro::expand(identifierToken &source) {
      bool isDefined = !!pp.getMacro(source.value);
      pp.push(new primitiveToken(source.origin,
                                 isDefined,
                                 isDefined ? "true" : "false"));
    }

    // __has_include()
    hasIncludeMacro::hasIncludeMacro(preprocessor &pp_) :
      macro_t(pp_, "__has_include") {
      argNames.add("INCLUDE_PATH", 0);
    }

    void hasIncludeMacro::expand(identifierToken &source) {
      // Make sure we have 3 tokens
      tokenVector tokens;
      token_t *token;
      for (int i = 0; i < 3; ++i) {
        if (pp.inputIsEmpty()) {
          pp.freeTokenVector(tokens);
          source.printError("Unfinished directive");
          return;
        }
        tokens.push_back(pp.getSourceToken());
      }

      std::string header;

      // Test for opening (
      token = tokens[0];
      bool hasError = true;
      if (token->type() & tokenType::op) {
        operatorToken &opToken = token->to<operatorToken>();
        hasError = (opToken.op.opType & operatorType::parenthesesStart);
      }
      if (hasError) {
        token->printError("Expected a header argument, like (\"header-path\")");
        pp.freeTokenVector(tokens);
        return;
      }

      // Test for string
      token = tokens[1];
      if (token->type() & tokenType::string) {
        header = token->to<stringToken>().value;
      } else {
        token->printError("Expected a string with the header path");
        pp.freeTokenVector(tokens);
        return;
      }

      // Test for closing )
      token = tokens[2];
      hasError = true;
      if (token->type() & tokenType::op) {
        operatorToken &opToken = token->to<operatorToken>();
        hasError = (opToken.op.opType & operatorType::parenthesesEnd);
      }
      if (hasError) {
        token->printError("Expected a closing )");
        pp.freeTokenVector(tokens);
        return;
      }

      bool hasInclude = io::exists(header);
      pp.push(new primitiveToken(source.origin,
                                 hasInclude,
                                 hasInclude ? "true" : "false"));
    }

    // __FILE__
    fileMacro::fileMacro(preprocessor &pp_) :
      macro_t(pp_, "__FILE__") {}

    void fileMacro::expand(identifierToken &source) {
      pp.push(new stringToken(source.origin,
                              source.origin.file->filename));
    }

    // __LINE__
    lineMacro::lineMacro(preprocessor &pp_) :
      macro_t(pp_, "__LINE__") {}

    void lineMacro::expand(identifierToken &source) {
      const int line = source.origin.position.line;
      pp.push(new primitiveToken(source.origin,
                                 line,
                                 occa::toString(line)));
    }

    // __DATE__
    dateMacro::dateMacro(preprocessor &pp_) :
      macro_t(pp_, "__DATE__") {}

    void dateMacro::expand(identifierToken &source) {
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

      pp.push(new stringToken(source.origin,
                              ss.str()));
    }

    // __TIME__
    timeMacro::timeMacro(preprocessor &pp_) :
      macro_t(pp_, "__TIME__") {}

    void timeMacro::expand(identifierToken &source) {
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

      pp.push(new stringToken(source.origin,
                              ss.str()));
    }

    // __COUNTER__
    counterMacro::counterMacro(preprocessor &pp_) :
      macro_t(pp_, "__COUNTER__"),
      counter(0) {}

    void counterMacro::expand(identifierToken &source) {
      const int value = counter;
      pp.push(new primitiveToken(source.origin,
                                 value,
                                 occa::toString(value)));
    }
  }
}
