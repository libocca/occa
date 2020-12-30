#ifndef OCCA_INTERNAL_LANG_TOKEN_QUALIFIERTOKEN_HEADER
#define OCCA_INTERNAL_LANG_TOKEN_QUALIFIERTOKEN_HEADER

#include <occa/internal/lang/token/token.hpp>

namespace occa {
  namespace lang {
    class qualifierToken : public token_t {
    public:
      const qualifier_t &qualifier;

      qualifierToken(const fileOrigin &origin_,
                     const qualifier_t &qualifier_);

      virtual ~qualifierToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
  }
}

#endif
