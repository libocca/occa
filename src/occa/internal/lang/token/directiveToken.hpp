#ifndef OCCA_INTERNAL_LANG_TOKEN_DIRECTIVETOKEN_HEADER
#define OCCA_INTERNAL_LANG_TOKEN_DIRECTIVETOKEN_HEADER

#include <occa/internal/lang/token/token.hpp>

namespace occa {
  namespace lang {
    class directiveToken : public token_t {
    public:
      std::string value;

      directiveToken(const fileOrigin &origin_,
                     const std::string &value_);

      virtual ~directiveToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
  }
}

#endif
