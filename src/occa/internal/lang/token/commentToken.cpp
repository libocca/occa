#include <occa/internal/lang/token/commentToken.hpp>

namespace occa {
  namespace lang {
    commentToken::commentToken(const fileOrigin &origin_,
                               const std::string &value_,
                               const int spacingType_) :
        token_t(origin_),
        value(value_),
        spacingType(spacingType_) {}

    commentToken::~commentToken() {}

    int commentToken::type() const {
      return tokenType::comment;
    }

    token_t* commentToken::clone() const {
      return new commentToken(origin, value, spacingType);
    }

    void commentToken::print(io::output &out) const {
      out << value;
    }
  }
}
