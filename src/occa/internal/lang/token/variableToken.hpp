#ifndef OCCA_INTERNAL_LANG_TOKEN_VARIABLETOKEN_HEADER
#define OCCA_INTERNAL_LANG_TOKEN_VARIABLETOKEN_HEADER

#include <occa/internal/lang/token/token.hpp>

namespace occa {
  namespace lang {
    class variableToken : public token_t {
    public:
      variable_t &value;

      variableToken(const fileOrigin &origin_,
                    variable_t &variable);

      virtual ~variableToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
  }
}

#endif
