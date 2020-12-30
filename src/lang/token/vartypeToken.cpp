#include <occa/internal/lang/token/vartypeToken.hpp>

namespace occa {
  namespace lang {
    vartypeToken::vartypeToken(const fileOrigin &origin_,
                               const vartype_t &value_) :
      token_t(origin_),
      value(value_) {}

    vartypeToken::~vartypeToken() {}

    int vartypeToken::type() const {
      return tokenType::vartype;
    }

    token_t* vartypeToken::clone() const {
      return new vartypeToken(origin, value);
    }

    void vartypeToken::print(io::output &out) const {
      out << value;
    }
  }
}
