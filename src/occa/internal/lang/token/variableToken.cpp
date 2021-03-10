#include <occa/internal/lang/token/variableToken.hpp>
#include <occa/internal/lang/variable.hpp>

namespace occa {
  namespace lang {
    variableToken::variableToken(const fileOrigin &origin_,
                                 variable_t &variable) :
      token_t(origin_),
      value(variable) {}

    variableToken::~variableToken() {}

    int variableToken::type() const {
      return tokenType::variable;
    }

    token_t* variableToken::clone() const {
      return new variableToken(origin, value);
    }

    void variableToken::print(io::output &out) const {
      out << value.name();
    }
  }
}
