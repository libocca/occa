#include <occa/internal/lang/token/stringToken.hpp>
#include <occa/internal/utils/string.hpp>

namespace occa {
  namespace lang {
    stringToken::stringToken(const fileOrigin &origin_,
                             const std::string &value_) :
      token_t(origin_),
      encoding(encodingType::none),
      value(value_),
      udf() {}

    stringToken::stringToken(const fileOrigin &origin_,
                             int encoding_,
                             const std::string &value_,
                             const std::string &udf_) :
      token_t(origin_),
      encoding(encoding_),
      value(value_),
      udf(udf_) {}

    stringToken::~stringToken() {}

    int stringToken::type() const {
      return tokenType::string;
    }

    token_t* stringToken::clone() const {
      return new stringToken(origin, encoding, value, udf);
    }

    void stringToken::append(const stringToken &token) {
      origin.position.end = token.origin.position.end;

      encoding = tokenType::mergeEncodings(encoding,
                                           token.encoding);
      value += token.value;
      udf = token.udf;
    }

    void stringToken::print(io::output &out) const {
      if (encoding & encodingType::ux) {
        if (encoding & encodingType::u8) {
          out << "u8";
        } else if (encoding & encodingType::u) {
          out << 'u';
        } else if (encoding & encodingType::U) {
          out << 'U';
        } else if (encoding & encodingType::L) {
          out << 'L';
        }
      }
      if (encoding & encodingType::R) {
        out << 'R';
      }
      out << '"' << escape(value, '"') << '"' << udf;
    }
  }
}
