#ifndef OCCA_INTERNAL_LANG_TOKEN_STRINGTOKEN_HEADER
#define OCCA_INTERNAL_LANG_TOKEN_STRINGTOKEN_HEADER

#include <occa/internal/lang/token/token.hpp>

namespace occa {
  namespace lang {
    class stringToken : public token_t {
    public:
      int encoding;
      std::string value;
      std::string udf;

      stringToken(const fileOrigin &origin_,
                  const std::string &value_);

      stringToken(const fileOrigin &origin_,
                  int encoding_,
                  const std::string &value_,
                  const std::string &udf_);

      virtual ~stringToken();

      virtual int type() const;

      virtual token_t* clone() const;

      void append(const stringToken &token);

      virtual void print(io::output &out) const;
    };
  }
}

#endif
