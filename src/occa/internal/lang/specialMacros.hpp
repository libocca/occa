#ifndef OCCA_INTERNAL_LANG_SPECIALMACROS_HEADER
#define OCCA_INTERNAL_LANG_SPECIALMACROS_HEADER

#include <occa/internal/lang/macro.hpp>

namespace occa {
  namespace lang {
    // defined()
    class definedMacro : public macro_t {
    public:
      definedMacro(preprocessor_t &pp_);

      virtual macro_t& clone(preprocessor_t &pp_) const;

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);
    };

    // __has_include()
    class hasIncludeMacro : public macro_t {
    public:
      hasIncludeMacro(preprocessor_t &pp_);

      virtual macro_t& clone(preprocessor_t &pp_) const;

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);
    };

    // __FILE__
    class fileMacro : public macro_t {
    public:
      fileMacro(preprocessor_t &pp_);

      virtual macro_t& clone(preprocessor_t &pp_) const;

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);
    };

    // __LINE__
    class lineMacro : public macro_t {
    public:
      lineMacro(preprocessor_t &pp_);

      virtual macro_t& clone(preprocessor_t &pp_) const;

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);
    };

    // __DATE__
    class dateMacro : public macro_t {
    public:
      dateMacro(preprocessor_t &pp_);

      virtual macro_t& clone(preprocessor_t &pp_) const;

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);
    };

    // __TIME__
    class timeMacro : public macro_t {
    public:
      timeMacro(preprocessor_t &pp_);

      virtual macro_t& clone(preprocessor_t &pp_) const;

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);
    };

    // __COUNTER__
    class counterMacro : public macro_t {
    public:
      mutable int counter;

      counterMacro(preprocessor_t &pp_,
                   const int counter_ = 0);

      virtual macro_t& clone(preprocessor_t &pp_) const;

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);
    };

    // OKL
    class oklMacro : public macro_t {
    public:
      oklMacro(preprocessor_t &pp_);

      virtual macro_t& clone(preprocessor_t &pp_) const;

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);
    };
  }
}

#endif
