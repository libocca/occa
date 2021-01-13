#ifndef OCCA_INTERNAL_LANG_TOKEN_FUNCTIONTOKEN_HEADER
#define OCCA_INTERNAL_LANG_TOKEN_FUNCTIONTOKEN_HEADER

#include <occa/internal/lang/token/token.hpp>

namespace occa {
  namespace lang {
    class functionToken : public token_t {
    public:
      function_t &value;

      functionToken(const fileOrigin &origin_,
                    function_t &function);

      virtual ~functionToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
  }
}

#endif
