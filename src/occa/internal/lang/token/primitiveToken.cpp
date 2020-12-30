#include <occa/internal/lang/token/primitiveToken.hpp>

namespace occa {
  namespace lang {
    primitiveToken::primitiveToken(const fileOrigin &origin_,
                                   const primitive &value_,
                                   const std::string &strValue_) :
      token_t(origin_),
      value(value_),
      strValue(strValue_) {}

    primitiveToken::~primitiveToken() {}

    int primitiveToken::type() const {
      return tokenType::primitive;
    }

    token_t* primitiveToken::clone() const {
      return new primitiveToken(origin, value, strValue);
    }

    void primitiveToken::print(io::output &out) const {
      out << strValue;
    }
  }
}
