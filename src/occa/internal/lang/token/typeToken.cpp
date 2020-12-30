#include <occa/internal/lang/token/typeToken.hpp>

namespace occa {
  namespace lang {
    typeToken::typeToken(const fileOrigin &origin_,
                         type_t &type_) :
      token_t(origin_),
      value(type_) {}

    typeToken::~typeToken() {}

    int typeToken::type() const {
      return tokenType::type;
    }

    token_t* typeToken::clone() const {
      return new typeToken(origin, value);
    }

    void typeToken::print(io::output &out) const {
      out << value.name();
    }
  }
}
