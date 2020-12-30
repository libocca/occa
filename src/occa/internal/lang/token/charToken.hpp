#ifndef OCCA_INTERNAL_LANG_TOKEN_CHARTOKEN_HEADER
#define OCCA_INTERNAL_LANG_TOKEN_CHARTOKEN_HEADER

#include <occa/internal/lang/token/token.hpp>

namespace occa {
  namespace lang {
    class charToken : public token_t {
    public:
      int encoding;
      std::string value;
      std::string udf;

      charToken(const fileOrigin &origin_,
                int encoding_,
                const std::string &value_,
                const std::string &udf_);

      virtual ~charToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
  }
}

#endif
