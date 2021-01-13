#include <occa/internal/lang/token/unknownToken.hpp>

namespace occa {
  namespace lang {
    unknownToken::unknownToken(const fileOrigin &origin_) :
      token_t(origin_) {}

    unknownToken::~unknownToken() {}

    int unknownToken::type() const {
      return tokenType::unknown;
    }

    token_t* unknownToken::clone() const {
      return new unknownToken(origin);
    }

    void unknownToken::print(io::output &out) const {
      out << origin.position.start[0];
    }
  }
}
