#include <sstream>
#include <ctime>

#include <occa/internal/utils/string.hpp>
#include <occa/internal/io.hpp>

#include <occa/internal/lang/preprocessor.hpp>
#include <occa/internal/lang/specialMacros.hpp>

namespace occa {
  namespace lang {
    // defined()
    definedMacro::definedMacro(preprocessor_t &pp_) :
      macro_t(pp_, "defined") {
      isFunctionLike = true;
      argNames["MACRO_NAME"] = 0;
    }

    macro_t& definedMacro::clone(preprocessor_t &pp_) const {
      return *(new definedMacro(pp_));
    }

    void definedMacro::expand(tokenVector &tokens,
                              identifierToken &source) {
      std::vector<tokenVector> allArgs;
      pp.expandingMacros = false;
      if (!loadArgs(source, allArgs) ||
          !checkArgs(source, allArgs)) {
        pp.expandingMacros = true;
        return;
      }
      pp.expandingMacros = true;

      // We only expect 1 argument
      tokenVector &args = allArgs[0];
      if (args.size() > 1) {
        args[1]->origin
          .from(false, thisToken.origin)
          .printError("Expected one macro name");

        freeTokenVector(args);
        return;
      }

      token_t *token = args[0];
      if (!(token->type() & tokenType::identifier)) {
        token->origin
          .from(false, thisToken.origin)
          .printError("Expected a token name identifier");

        freeTokenVector(args);
        return;
      }

      bool isDefined = !!pp.getMacro(token->to<identifierToken>().value);
      tokens.push_back(new primitiveToken(source.origin,
                                          isDefined,
                                          isDefined ? "true" : "false"));

      freeTokenVector(args);
    }

    // __has_include()
    hasIncludeMacro::hasIncludeMacro(preprocessor_t &pp_) :
      macro_t(pp_, "__has_include") {
      isFunctionLike = true;
      argNames["INCLUDE_PATH"] = 0;
    }

    macro_t& hasIncludeMacro::clone(preprocessor_t &pp_) const {
      return *(new hasIncludeMacro(pp_));
    }

    void hasIncludeMacro::expand(tokenVector &tokens,
                                 identifierToken &source) {
      std::vector<tokenVector> allArgs;
      if (!loadArgs(source, allArgs) ||
          !checkArgs(source, allArgs)) {
        return;
      }

      // We only expect 1 argument
      tokenVector &args = allArgs[0];

      // Extract the header
      const int tokenCount = (int) args.size();
      for (int i = 0; i < tokenCount; ++i) {
        if (!(args[i]->type() & tokenType::string)) {
          args[i]->printError("Expected a string with the header path");
          freeTokenVector(tokens);
          return;
        }
      }
      // Append string tokens
      stringToken &strToken = args[0]->to<stringToken>();
      for (int i = 1; i < tokenCount; ++i) {
        strToken.append(args[i]->to<stringToken>());
        delete args[i];
      }

      bool hasInclude = io::exists(strToken.value);
      delete &strToken;

      tokens.push_back(new primitiveToken(source.origin,
                                          hasInclude,
                                          hasInclude ? "true" : "false"));
    }

    // __FILE__
    fileMacro::fileMacro(preprocessor_t &pp_) :
      macro_t(pp_, "__FILE__") {}

    macro_t& fileMacro::clone(preprocessor_t &pp_) const {
      return *(new fileMacro(pp_));
    }

    void fileMacro::expand(tokenVector &tokens,
                           identifierToken &source) {
      tokens.push_back(new stringToken(source.origin,
                                       source.origin.file->filename));
    }

    // __LINE__
    lineMacro::lineMacro(preprocessor_t &pp_) :
      macro_t(pp_, "__LINE__") {}

    macro_t& lineMacro::clone(preprocessor_t &pp_) const {
      return *(new lineMacro(pp_));
    }

    void lineMacro::expand(tokenVector &tokens,
                           identifierToken &source) {
      const int line = source.origin.position.line;
      tokens.push_back(new primitiveToken(source.origin,
                                          line,
                                          occa::toString(line)));
    }

    // __DATE__
    dateMacro::dateMacro(preprocessor_t &pp_) :
      macro_t(pp_, "__DATE__") {}

    macro_t& dateMacro::clone(preprocessor_t &pp_) const {
      return *(new dateMacro(pp_));
    }

    void dateMacro::expand(tokenVector &tokens,
                           identifierToken &source) {
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

      tokens.push_back(new stringToken(source.origin,
                                       ss.str()));
    }

    // __TIME__
    timeMacro::timeMacro(preprocessor_t &pp_) :
      macro_t(pp_, "__TIME__") {}

    macro_t& timeMacro::clone(preprocessor_t &pp_) const {
      return *(new timeMacro(pp_));
    }

    void timeMacro::expand(tokenVector &tokens,
                           identifierToken &source) {
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

      tokens.push_back(new stringToken(source.origin,
                                       ss.str()));
    }

    // __COUNTER__
    counterMacro::counterMacro(preprocessor_t &pp_,
                               const int counter_) :
      macro_t(pp_, "__COUNTER__"),
      counter(counter_) {}

    macro_t& counterMacro::clone(preprocessor_t &pp_) const {
      return *(new counterMacro(pp_, counter));
    }

    void counterMacro::expand(tokenVector &tokens,
                              identifierToken &source) {
      const int value = counter++;
      tokens.push_back(new primitiveToken(source.origin,
                                          value,
                                          occa::toString(value)));
    }

    // OKL
    oklMacro::oklMacro(preprocessor_t &pp_) :
      macro_t(pp_, "OKL") {
      isFunctionLike = true;
      argNames["SOURCE_CODE"] = 0;
    }

    macro_t& oklMacro::clone(preprocessor_t &pp_) const {
      return *(new oklMacro(pp_));
    }

    void oklMacro::expand(tokenVector &tokens,
                          identifierToken &source) {
      std::vector<tokenVector> allArgs;
      pp.expandingMacros = false;
      if (!loadArgs(source, allArgs) ||
          !checkArgs(source, allArgs)) {
        pp.expandingMacros = true;
        return;
      }
      pp.expandingMacros = true;

      if (!allArgs.size()) {
        pp.errorOn(&source, "Expected source code");
        return;
      }

      // We only expect 1 argument
      tokenVector &args = allArgs[0];
      if (args.size() > 1) {
        args[1]->origin
          .from(false, thisToken.origin)
          .printError("Expected only one source code string");

        freeTokenVector(args);
        return;
      }

      token_t *token = args[0];
      if (!(token->type() & tokenType::string)) {
        token->origin
          .from(false, thisToken.origin)
          .printError("Expected a source code string");

        freeTokenVector(args);
        return;
      }

      // Check if the next argument is an empty semicolon to remove
      //
      // Example:
      //   OKL("@inner");
      //   for(...)
      // ->
      //   @inner
      //   for(...)
      //
      token_t *nextToken = pp.getSourceToken();
      if (nextToken) {
        if (token_t::safeOperatorType(nextToken) == operatorType::semicolon) {
          delete nextToken;
        } else {
          pp.pushInput(nextToken);
        }
      }

      const std::string &content = token->to<stringToken>().value;
      pp.injectSourceCode(*token, strip(content));

    }
  }
}
