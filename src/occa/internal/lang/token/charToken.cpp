#include <occa/internal/lang/token/charToken.hpp>
#include <occa/internal/utils/string.hpp>

namespace occa {
  namespace lang {
    charToken::charToken(const fileOrigin &origin_,
                         int encoding_,
                         const std::string &value_,
                         const std::string &udf_) :
      token_t(origin_),
      encoding(encoding_),
      value(value_),
      udf(udf_) {}

    charToken::~charToken() {}

    int charToken::type() const {
      return tokenType::char_;
    }

    token_t* charToken::clone() const {
      return new charToken(origin, encoding, value, udf);
    }

    void charToken::print(io::output &out) const {
      if (encoding & encodingType::u) {
        out << 'u';
      } else if (encoding & encodingType::U) {
        out << 'U';
      } else if (encoding & encodingType::L) {
        out << 'L';
      }
      out << '\'' << escape(value, '\'') << '\'' << udf;
    }
  }
}
