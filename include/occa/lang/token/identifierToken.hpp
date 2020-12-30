#ifndef OCCA_INTERNAL_LANG_TOKEN_IDENTIFIERTOKEN_HEADER
#define OCCA_INTERNAL_LANG_TOKEN_IDENTIFIERTOKEN_HEADER

#include <occa/internal/lang/token/token.hpp>

namespace occa {
  namespace lang {
    class identifierToken : public token_t {
    public:
      std::string value;

      identifierToken(const fileOrigin &origin_,
                      const std::string &value_);

      virtual ~identifierToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
  }
}

#endif
