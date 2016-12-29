class macro_t {
public:
  std::string name;
  bool isAFunction, hasVarArgs;

  int argc;
  std::vector<std::string> parts;
  std::vector<int> argBetweenParts;

  int definedLine, undefinedLine;
};