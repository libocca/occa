#include <occa/internal/lang/statement/commentStatement.hpp>
#include <occa/internal/utils/string.hpp>

namespace occa {
  namespace lang {
    commentStatement::commentStatement(blockStatement *up_,
                                       const commentToken &token_) :
        statement_t(up_, &token_),
        token((commentToken&) *source) {}

    commentStatement::commentStatement(blockStatement *up_,
                                       const commentStatement &other) :
        statement_t(up_, other),
        token((commentToken&) *source) {}

    commentStatement::~commentStatement() {}

    statement_t& commentStatement::clone_(blockStatement *up_) const {
      return *(new commentStatement(up_, *this));
    }

    int commentStatement::type() const {
      return statementType::comment;
    }

    std::string commentStatement::statementName() const {
      return "comment";
    }

    void commentStatement::print(printer &pout) const {
      strVector lines = split(token.value, '\n');
      const int lineCount = lines.size();

      pout.printEndNewline();

      if (token.spacingType & spacingType_t::left) {
        pout << '\n';
      }

      for (int i = 0; i < lineCount; ++i) {
        pout.printIndentation();
        pout << strip(lines[i]);
        pout.printNewline();
      }

      if (token.spacingType & spacingType_t::right) {
        pout << '\n';
      }
    }
  }
}
