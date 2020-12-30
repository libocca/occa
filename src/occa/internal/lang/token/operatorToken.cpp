#include <occa/internal/lang/token/operatorToken.hpp>

namespace occa {
  namespace lang {
    operatorToken::operatorToken(const fileOrigin &origin_,
                                 const operator_t &op_) :
      token_t(origin_),
      op(&op_) {}

    operatorToken::~operatorToken() {}

    int operatorToken::type() const {
      return tokenType::op;
    }

    const opType_t& operatorToken::opType() const {
      return op->opType;
    }

    token_t* operatorToken::clone() const {
      return new operatorToken(origin, *op);
    }

    void operatorToken::print(io::output &out) const {
      out << op->str;
    }
  }
}
