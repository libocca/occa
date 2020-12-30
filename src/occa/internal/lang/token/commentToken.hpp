#ifndef OCCA_INTERNAL_LANG_TOKEN_COMMENTTOKEN_HEADER
#define OCCA_INTERNAL_LANG_TOKEN_COMMENTTOKEN_HEADER

#include <occa/internal/lang/token/token.hpp>

namespace occa {
  namespace lang {
    namespace spacingType_t {
      static const int none  = 0;
      static const int left  = (1 << 0);
      static const int right = (1 << 1);
    }

    class commentToken : public token_t {
    public:
      std::string value;
      int spacingType;

      commentToken(const fileOrigin &origin_,
                   const std::string &value_,
                   const int spacingType_);

      virtual ~commentToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
  }
}

#endif
