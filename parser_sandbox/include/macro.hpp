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
    macroPart_t(const std::string &str_);
  };
  //====================================

  //---[ Macro ]-------------------------
  class macro_t {
  public:
    static const std::string VA_ARGS;
    std::string name;

    int argc;
    bool hasVarArgs;
    macroPartVector_t parts;

    int definedLine, undefinedLine;

    macro_t();
    macro_t(const char *c);
    macro_t(const char *c, const int chars);
    macro_t(const std::string &s);

    void load(const std::string &s);
    void load(const char *c, const int chars);

    void loadName(const char *&c);
    void loadArgs(const char *&c, macroPartVector_t &argNames, const bool keepWhitespace = false) const;
    void setParts(const char *&c, macroPartVector_t &argNames);

    void clear();
    std::string expand() const;
    std::string expand(const char *c, const int chars) const;
    std::string expand(const std::string &s) const;
    std::string expand(const macroPartVector_t &args) const;
  };

  std::ostream& operator << (std::ostream &out, const macro_t &macro);
  //====================================
}

#endif
