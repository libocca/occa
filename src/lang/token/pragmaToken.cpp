#include <occa/internal/lang/token/pragmaToken.hpp>

namespace occa {
  namespace lang {
    pragmaToken::pragmaToken(const fileOrigin &origin_,
                             const std::string &value_) :
      token_t(origin_),
      value(value_) {}

    pragmaToken::~pragmaToken() {}

    int pragmaToken::type() const {
      return tokenType::pragma;
    }

    token_t* pragmaToken::clone() const {
      return new pragmaToken(origin, value);
    }

    void pragmaToken::print(io::output &out) const {
      out << "#pragma " << value << '\n';
    }
  }
}
