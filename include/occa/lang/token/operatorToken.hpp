#ifndef OCCA_INTERNAL_LANG_TOKEN_OPERATORTOKEN_HEADER
#define OCCA_INTERNAL_LANG_TOKEN_OPERATORTOKEN_HEADER

#include <occa/internal/lang/token/token.hpp>

namespace occa {
  namespace lang {
    class operatorToken : public token_t {
    public:
      const operator_t *op;

      operatorToken(const fileOrigin &origin_,
                    const operator_t &op_);

      virtual ~operatorToken();

      virtual int type() const;

      virtual const opType_t& opType() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
  }
}

#endif
