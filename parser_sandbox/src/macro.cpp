#include <cstring>

#include "occa/defines.hpp"
#include "occa/types.hpp"
#include "occa/tools/lex.hpp"
#include "occa/tools/string.hpp"
#include "occa/tools/sys.hpp"

#include "macro.hpp"
#include "preprocessor.hpp"

namespace occa {
  //---[ Part ]-------------------------
  macroPart_t::macroPart_t(const int info_) :
    info(info_) {}

  macroPart_t::macroPart_t(const char *c) :
    info(macroInfo::string),
    str(c) {}

  macroPart_t::macroPart_t(const std::string &str_) :
    info(macroInfo::string),
    str(str_) {}
  //====================================

  //---[ Macro ]-------------------------
  const std::string macro_t::VA_ARGS = "__VA_ARGS__";

  macro_t::macro_t(const preprocessor_t *preprocessor_) :
    preprocessor(preprocessor_),
    macroStart(NULL),
    definedLine(-1),
    undefinedLine(-1) {}

  macro_t::macro_t(const preprocessor_t *preprocessor_, char *c) :
    preprocessor(preprocessor_),
    definedLine(-1),
    undefinedLine(-1) {

    macroStart = c;
    load(c);
  }

  macro_t::macro_t(const preprocessor_t *preprocessor_, const char *c) :
    preprocessor(preprocessor_),
    definedLine(-1),
    undefinedLine(-1) {

    macroStart = c;
    std::string s(c);
    load(&(s[0]));
  }

  void macro_t::load(const char *c) {
    macroStart = c;
    std::string s(c);
    load(&(s[0]));
  }

  void macro_t::load(char *c) {
    clear();
    localMacroStart = c;

    loadName(c);

    if (*c != '(') {
      parts.push_back(c);
    } else {
      macroPartVector_t argNames;
      loadArgs(c, argNames);
      argc = (int) argNames.size();

      setParts(c, argNames);
    }

    if (c > localMacroStart) {
      source = "#define ";
      source += std::string(localMacroStart, c - localMacroStart);
    }
  }

  void macro_t::loadName(char *&c) {
    lex::skipWhitespace(c);

    if (*c == '\0') {
      printError(c, "Macro name missing");
      return;
    }
    if (!lex::charIsIn(*c, lex::identifierStartChar)) {
      printError(c, "Macro name must be an identifier: [a-zA-Z_]([a-zA-Z0-9_]*)");
      return;
    }

    char *nameStart = c++;
    lex::skipFrom(c, lex::identifierChars);
    name = std::string(nameStart, c - nameStart);

    if (!lex::isWhitespace(*c) && (*c != '(') && (*c != '\0')) {
      printWarning(c, "Whitespace recommended after macro name");
    } else {
      lex::skipWhitespace(c);
    }
  }

  void macro_t::loadArgs(char *&c, macroPartVector_t &argNames, const bool loadingArgNames) const {
    static std::string startDelimiters = std::string(lex::whitespaceChars) + "(";
    static std::string endDelimiters(",)");

    lex::skipTo(c, startDelimiters, '\\');
    if (*c != '(') {
      return;
    }

    // Skip '('
    ++c;
    lex::skipWhitespace(c);
    char *argsStart = c;
    lex::skipTo(c, ')');
    char *argsEnd = c;
    if (*argsEnd != ')') {
      printError(c, "Missing closing \")\"");
    }

    c = argsStart;
    while(c < argsEnd) {
      c += (*c == ',');
      char *start = c;
      lex::skipWhitespace(c);
      char *argStart = c;
      lex::quotedSkipTo(c, endDelimiters);
      char *argEnd = c;

      macroPart_t arg;
      arg.str = std::string(argStart, argEnd - argStart);
      if (loadingArgNames) {
        if (lex::isWhitespace(*start)) {
          arg.info |= macroInfo::hasLeftSpace;
        }
        if (lex::isWhitespace(*(argEnd - 1))) {
          arg.info |= macroInfo::hasRightSpace;
        }
      }

      if (loadingArgNames && hasVarArgs) {
        printFatalError(argStart, "Variable arguments (...) must be the last argument");
      }

      if (loadingArgNames && arg.str == "...") {
        hasVarArgs = true;
      } else {
        argNames.push_back(arg);
      }
    }
    c = argsEnd + (*argsEnd == ')');
  }

  void macro_t::setParts(char *&c, macroPartVector_t &argNames) {
    static std::string delimiters;
    // Setup word delimeters [a-zA-Z0-9]
    if (delimiters.size() == 0) {
      int pos = 0;
      delimiters.resize(26 + 26 + 10 + 1);
      for (char c_ = 'a'; c_ <= 'z'; ++c_) {
        delimiters[pos++] = c_;
        delimiters[pos++] = ('A' + c_ - 'a');
      }
      for (char c_ = '0'; c_ <= '9'; ++c_) {
        delimiters[pos++] = c_;
      }
      delimiters[pos++] = '_';
    }

    lex::skipWhitespace(c);
    char *cStart = c;
    while (*c != '\0') {
      lex::skipTo(c, delimiters);
      char *partStart = c;
      lex::skipFrom(c, delimiters);

      const int partSize = (c - partStart);
      macroPart_t part;
      // Iterate over argument names if part starts with [a-zA-Z0-9]
      if ((*partStart < '0') || ('9' < *partStart)) {
        for (int i = 0; i < argc; ++i) {
          const std::string &argName = argNames[i].str;
          if ((partSize != (int) argName.size()) ||
              strncmp(argName.c_str(), partStart, partSize)) {
            continue;
          }
          // Setup macro part
          part.argPos = i;
          part.info   = macroInfo::arg;
        }
        if (hasVarArgs                         &&
            (part.info == 0)                   &&
            (partSize == (int) VA_ARGS.size()) &&
            !strncmp(VA_ARGS.c_str(), partStart, partSize)) {
          part.argPos = -1;
          part.info = (macroInfo::arg | macroInfo::variadic);
        }
      }
      // Add lazy string part if argument was found
      if (part.info) {
        if (cStart < partStart) {
          const int strChars = (partStart - cStart);
          std::string str(cStart, strChars);

          // Change arguemnt type if needed
          if (str[strChars - 1] == '#') {
            if ((2 <= strChars) &&
                str[strChars - 2] == '#') {
              part.info |= macroInfo::concat;
              str = str.substr(0, strChars - 2);
            } else {
              part.info |= macroInfo::stringify;
              str = str.substr(0, strChars - 1);
            }
          }
          // Ignore strings only made with # or ##
          if (str.size()) {
            parts.push_back(str);
          }
        }
        // Push back argument part
        parts.push_back(part);
          // Update the lazy string start
        lex::skipWhitespace(c);
        cStart = c;
      }
    }
    if (cStart < c) {
      parts.push_back(std::string(cStart, c - cStart));
    }
  }

  void macro_t::clear() {
    name = "";
    source = "";

    argc = 0;
    hasVarArgs = false;
    parts.clear();

    definedLine   = -1;
    undefinedLine = -1;
  }

  std::string macro_t::expand(const char *c) const {
    std::string s(c);
    char *c2 = &(s[0]);
    return expand(c2);
  }

  std::string macro_t::expand(char *&c) const {
    const int partCount = (int) parts.size();

    if (partCount == 0) {
      return "";
    } else if ((argc == 0) && !hasVarArgs) {
      return parts[0].str;
    }

    macroPartVector_t args;
    loadArgs(c, args, false);

    const int inputArgc = (int) args.size();
    std::string ret;

    for (int i = 0; i < partCount; ++i) {
      const macroPart_t &part = parts[i];
      const size_t startRetSize = ret.size();

      if (part.info & macroInfo::string) {
        ret += part.str;
      } else if (part.info & macroInfo::arg) {
        std::string argStr;
        if (part.info & macroInfo::variadic) {
          for (int j = argc; j < inputArgc; ++j) {
            // Only add a space if there is there is an argument
            if ((argc < j) &&
                ((args[j].info & macroInfo::hasSpace) ||
                 args[j].str.size())) {
              argStr += ' ';
            }
            argStr += args[j].str;
            // ##__VA_ARGS__ doesn't print trailing,
            if ((j < (inputArgc - 1)) &&
                ((j < (inputArgc - 2))            ||
                 !(part.info & macroInfo::concat) ||
                 (0 < args[j + 1].str.size()))) {
                argStr += ',';
            }
          }
        } else if (part.argPos < inputArgc) {
          argStr = args[part.argPos].str;
        }

        // Modify argStr based on stringify, concat, and spaces
        if (part.info & macroInfo::stringify) {
          ret += '"';
          ret += argStr;
          ret += '"';
        } else if (part.info & macroInfo::concat) {
          ret += argStr;
        } else {
          if ((part.argPos < 0) || (inputArgc <= part.argPos)) {
            ret += argStr;
          } else {
            const macroPart_t arg = args[part.argPos];
            if (arg.info & macroInfo::hasLeftSpace) {
              ret += ' ';
            }
            ret += argStr;
            if (arg.info & macroInfo::hasRightSpace) {
              ret += ' ';
            }
          }
        }
      }
      // Separate inputs with spaces
      if ((i < (partCount - 1))                    &&
          !(parts[i + 1].info & macroInfo::concat) &&
          (ret.size() != startRetSize)) {
        ret += ' ';
      }
    }

    return ret;
  }

  std::string macro_t::toString() const {
    return source;
  }

  macro_t::operator std::string() const {
    return source;
  }

  void macro_t::print() const {
    std::cout << toString();
  }

  //  ---[ Messages ]-------------------
  void macro_t::printError(const char *c,
                           const std::string &message) const {
    preprocessor->printError(message, macroStart + (c - localMacroStart));
  }

  void macro_t::printFatalError(const char *c,
                                const std::string &message) const {
    preprocessor->printFatalError(message, macroStart + (c - localMacroStart));
  }

  void macro_t::printWarning(const char *c,
                             const std::string &message) const {
    preprocessor->printWarning(message, macroStart + (c - localMacroStart));
  }
  //  ==================================
  //====================================
}
