#ifndef OCCA_LANG_TOKEN_COMMENTTOKEN_HEADER
#define OCCA_LANG_TOKEN_COMMENTTOKEN_HEADER

#include <occa/lang/token/token.hpp>

namespace occa {
  namespace lang {
    class commentToken : public token_t {
    public:
      std::string value;

      commentToken(const fileOrigin &origin_,
                   const std::string &value_);

      virtual ~commentToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
  }
}

#endif
