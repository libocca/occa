#ifndef OCCA_INTERNAL_LANG_TOKEN_VARTYPETOKEN_HEADER
#define OCCA_INTERNAL_LANG_TOKEN_VARTYPETOKEN_HEADER

#include <occa/internal/lang/token/token.hpp>

namespace occa {
  namespace lang {
    class vartypeToken : public token_t {
    public:
      vartype_t value;

      vartypeToken(const fileOrigin &origin_,
                   const vartype_t &value_);

      virtual ~vartypeToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
  }
}

#endif
