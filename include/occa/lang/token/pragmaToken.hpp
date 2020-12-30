#ifndef OCCA_INTERNAL_LANG_TOKEN_PRAGMATOKEN_HEADER
#define OCCA_INTERNAL_LANG_TOKEN_PRAGMATOKEN_HEADER

#include <occa/internal/lang/token/token.hpp>

namespace occa {
  namespace lang {
    class pragmaToken : public token_t {
     public:
      std::string value;

      pragmaToken(const fileOrigin &origin_,
                  const std::string &value_);

      virtual ~pragmaToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
  }
}

#endif
