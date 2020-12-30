#include <occa/internal/lang/token/directiveToken.hpp>

namespace occa {
  namespace lang {
    directiveToken::directiveToken(const fileOrigin &origin_,
                                   const std::string &value_) :
      token_t(origin_),
      value(value_) {}

    directiveToken::~directiveToken() {}

    int directiveToken::type() const {
      return tokenType::directive;
    }

    token_t* directiveToken::clone() const {
      return new directiveToken(origin, value);
    }

    void directiveToken::print(io::output &out) const {
      out << '#' << value << '\n';
    }
  }
}
