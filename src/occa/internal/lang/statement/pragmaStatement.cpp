#include <occa/internal/lang/statement/pragmaStatement.hpp>

namespace occa {
  namespace lang {
    pragmaStatement::pragmaStatement(blockStatement *up_,
                                     const pragmaToken &token_) :
      statement_t(up_, &token_),
      token((pragmaToken&) *source) {}

    pragmaStatement::pragmaStatement(blockStatement *up_,
                                     const pragmaStatement &other) :
      statement_t(up_, other),
      token((pragmaToken&) *source) {}

    pragmaStatement::~pragmaStatement() {}

    statement_t& pragmaStatement::clone_(blockStatement *up_) const {
      return *(new pragmaStatement(up_, *this));
    }

    int pragmaStatement::type() const {
      return statementType::pragma;
    }

    std::string pragmaStatement::statementName() const {
      return "pragma";
    }

    std::string& pragmaStatement::value() {
      return token.value;
    }

    const std::string& pragmaStatement::value() const {
      return token.value;
    }

    void pragmaStatement::print(printer &pout) const {
      pout << "#pragma " << token.value << '\n';
    }
  }
}
