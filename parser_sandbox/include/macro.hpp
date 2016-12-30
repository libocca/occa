#ifndef OCCA_PARSER_MACRO_HEADER2
#define OCCA_PARSER_MACRO_HEADER2

#include <vector>

class macro_t {
public:
  std::string name;
  bool isAFunction, hasVarArgs;

  int argc;
  std::vector<std::string> parts;
  std::vector<int> argBetweenParts;

  int definedLine, undefinedLine;
};

#endif
