#ifndef OCCA_INTERNAL_LANG_TOKEN_UNKNOWNTOKEN_HEADER
#define OCCA_INTERNAL_LANG_TOKEN_UNKNOWNTOKEN_HEADER

#include <occa/internal/lang/token/token.hpp>

namespace occa {
  namespace lang {
    class unknownToken : public token_t {
    public:
      char symbol;

      unknownToken(const fileOrigin &origin_);

      virtual ~unknownToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
  }
}

#endif
