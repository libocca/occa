#ifndef OCCA_PARSER_MACRO_HEADER2
#define OCCA_PARSER_MACRO_HEADER2

#include <vector>

namespace occa {
  class macroPart_t;
  typedef std::vector<macroPart_t> macroPartVector_t;

  namespace macroInfo {
    static const int string        = (1 << 0);
    static const int arg           = (1 << 1);
    static const int stringify     = (1 << 2);
    static const int concat        = (1 << 3);
    static const int variadic      = (1 << 4);

    static const int hasSpace      = (3 << 5);
    static const int hasLeftSpace  = (1 << 5);
    static const int hasRightSpace = (1 << 6);
  }

  //---[ Part ]-------------------------
  class macroPart_t {
  public:
    int info;
    std::string str;
    int argPos;

    macroPart_t(const int info_ = 0);
    macroPart_t(const char *c);
    macroPart_t(const std::string &str_);
  };
  //====================================

  //---[ Macro ]-------------------------
  class preprocessor_t;

  class macro_t {
  public:
    static const std::string VA_ARGS;

    const preprocessor_t *preprocessor;
    const char *macroStart;
    const char *localMacroStart;

    std::string name, source;

    int argc;
    mutable bool hasVarArgs;
    macroPartVector_t parts;

    int definedLine, undefinedLine;

    macro_t(const preprocessor_t *preprocessor_ = NULL);
    macro_t(const preprocessor_t *preprocessor_, char *c);
    macro_t(const preprocessor_t *preprocessor_, const char *c);

    void load(char *c);
    void load(const char *c);

    void loadName(char *&c);
    void loadArgs(char *&c, macroPartVector_t &argNames, const bool loadingArgNames = true) const;
    void setParts(char *&c, macroPartVector_t &argNames);

    void clear();
    std::string expand(const char *c) const;
    virtual std::string expand(char *&c) const;

    std::string toString() const;
    operator std::string() const;
    void print() const;

    //---[ Messages ]-------------------
    void printError(const char *c, const std::string &message) const;
    void printFatalError(const char *c, const std::string &message) const;
    void printWarning(const char *c, const std::string &message) const;
    //==================================
  };
  //====================================
}

#endif
