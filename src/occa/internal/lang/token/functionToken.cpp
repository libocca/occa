#include <occa/internal/lang/token/functionToken.hpp>

namespace occa {
  namespace lang {
    functionToken::functionToken(const fileOrigin &origin_,
                                 function_t &function) :
      token_t(origin_),
      value(function) {}

    functionToken::~functionToken() {}

    int functionToken::type() const {
      return tokenType::function;
    }

    token_t* functionToken::clone() const {
      return new functionToken(origin, value);
    }

    void functionToken::print(io::output &out) const {
      out << value.name();
    }
  }
}
