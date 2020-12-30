#ifndef OCCA_INTERNAL_LANG_TOKEN_NEWLINETOKEN_HEADER
#define OCCA_INTERNAL_LANG_TOKEN_NEWLINETOKEN_HEADER

#include <occa/internal/lang/token/token.hpp>

namespace occa {
  namespace lang {
    class newlineToken : public token_t {
    public:
      newlineToken(const fileOrigin &origin_);

      virtual ~newlineToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
  }
}

#endif
