#include <occa/internal/lang/token/identifierToken.hpp>

namespace occa {
  namespace lang {
    identifierToken::identifierToken(const fileOrigin &origin_,
                                     const std::string &value_) :
      token_t(origin_),
      value(value_) {}

    identifierToken::~identifierToken() {}

    int identifierToken::type() const {
      return tokenType::identifier;
    }

    token_t* identifierToken::clone() const {
      return new identifierToken(origin, value);
    }

    void identifierToken::print(io::output &out) const {
      out << value;
    }
  }
}
