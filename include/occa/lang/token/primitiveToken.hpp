#ifndef OCCA_INTERNAL_LANG_TOKEN_PRIMITIVETOKEN_HEADER
#define OCCA_INTERNAL_LANG_TOKEN_PRIMITIVETOKEN_HEADER

#include <occa/internal/lang/token/token.hpp>

namespace occa {
  namespace lang {
    class primitiveToken : public token_t {
    public:
      primitive value;
      std::string strValue;

      primitiveToken(const fileOrigin &origin_,
                     const primitive &value_,
                     const std::string &strValue_);

      virtual ~primitiveToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
  }
}

#endif
