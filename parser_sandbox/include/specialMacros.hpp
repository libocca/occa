#ifndef OCCA_PARSER_SPECIALMACROS_HEADER2
#define OCCA_PARSER_SPECIALMACROS_HEADER2

#include "macro.hpp"

namespace occa {
  // __FILE__
  class fileMacro_t : public macro_t {
  public:
    fileMacro_t(const preprocessor_t *preprocessor_);
    std::string expand(char *&c) const;
  };

  // __LINE__
  class lineMacro_t : public macro_t {
  public:
    lineMacro_t(const preprocessor_t *preprocessor_);
    std::string expand(char *&c) const;
  };

  // __DATE__
  class dateMacro_t : public macro_t {
  public:
    dateMacro_t(const preprocessor_t *preprocessor_);
    std::string expand(char *&c) const;
  };

  // __TIME__
  class timeMacro_t : public macro_t {
  public:
    timeMacro_t(const preprocessor_t *preprocessor_);
    std::string expand(char *&c) const;
  };

  // __COUNTER__
  class counterMacro_t : public macro_t {
  public:
    mutable int counter;

    counterMacro_t(const preprocessor_t *preprocessor_);
    std::string expand(char *&c) const;
  };
}
#endif
