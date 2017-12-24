#ifndef OCCA_PARSER_SPECIALMACROS_HEADER2
#define OCCA_PARSER_SPECIALMACROS_HEADER2

#include "macro.hpp"

namespace occa {
  // __FILE__
  class fileMacro : public macro_t {
  public:
    fileMacro(const preprocessor_t *preprocessor_);
    std::string expand(char *&c) const;
  };

  // __LINE__
  class lineMacro : public macro_t {
  public:
    lineMacro(const preprocessor_t *preprocessor_);
    std::string expand(char *&c) const;
  };

  // __DATE__
  class dateMacro : public macro_t {
  public:
    dateMacro(const preprocessor_t *preprocessor_);
    std::string expand(char *&c) const;
  };

  // __TIME__
  class timeMacro : public macro_t {
  public:
    timeMacro(const preprocessor_t *preprocessor_);
    std::string expand(char *&c) const;
  };

  // __COUNTER__
  class counterMacro : public macro_t {
  public:
    mutable int counter;

    counterMacro(const preprocessor_t *preprocessor_);
    std::string expand(char *&c) const;
  };
}
#endif
