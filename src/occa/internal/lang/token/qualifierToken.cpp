#include <occa/internal/lang/token/qualifierToken.hpp>

namespace occa {
  namespace lang {
    qualifierToken::qualifierToken(const fileOrigin &origin_,
                                   const qualifier_t &qualifier_) :
      token_t(origin_),
      qualifier(qualifier_) {}

    qualifierToken::~qualifierToken() {}

    int qualifierToken::type() const {
      return tokenType::qualifier;
    }

    token_t* qualifierToken::clone() const {
      return new qualifierToken(origin, qualifier);
    }

    void qualifierToken::print(io::output &out) const {
      out << qualifier.name;
    }
  }
}
