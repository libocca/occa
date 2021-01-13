#include <occa/internal/lang/token/newlineToken.hpp>

namespace occa {
  namespace lang {
    newlineToken::newlineToken(const fileOrigin &origin_) :
      token_t(origin_) {}

    newlineToken::~newlineToken() {}

    int newlineToken::type() const {
      return tokenType::newline;
    }

    token_t* newlineToken::clone() const {
      return new newlineToken(origin);
    }

    void newlineToken::print(io::output &out) const {
      out << '\n';
    }
  }
}
