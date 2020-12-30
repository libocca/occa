#ifndef OCCA_INTERNAL_LANG_TOKEN_TYPETOKEN_HEADER
#define OCCA_INTERNAL_LANG_TOKEN_TYPETOKEN_HEADER

#include <occa/internal/lang/token/token.hpp>

namespace occa {
  namespace lang {
    class typeToken : public token_t {
    public:
      type_t &value;

      typeToken(const fileOrigin &origin_,
                type_t &type_);

      virtual ~typeToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
  }
}

#endif
