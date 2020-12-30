#include <occa/internal/lang/statement/sourceCodeStatement.hpp>
#include <occa/internal/lang/utils/array.hpp>
#include <occa/internal/utils/string.hpp>

namespace occa {
  namespace lang {
    sourceCodeStatement::sourceCodeStatement(blockStatement *up_,
                                             token_t *sourceToken,
                                             const std::string &sourceCode_) :
        statement_t(up_, sourceToken),
        sourceCode(sourceCode_) {}

    sourceCodeStatement::sourceCodeStatement(blockStatement *up_,
                                           const sourceCodeStatement &other) :
        statement_t(up_, other),
        sourceCode(other.sourceCode) {}

    sourceCodeStatement::~sourceCodeStatement() {}

    statement_t& sourceCodeStatement::clone_(blockStatement *up_) const {
      return *(new sourceCodeStatement(up_, *this));
    }

    int sourceCodeStatement::type() const {
      return statementType::sourceCode;
    }

    std::string sourceCodeStatement::statementName() const {
      return "source code";
    }

    void sourceCodeStatement::print(printer &pout) const {
      array<std::string> lines = split(sourceCode, '\n');

      lines.forEach([&](std::string line) {
          pout.printStartIndentation();
          pout << strip(line);
          pout.printEndNewline();
        });
    }
  }
}
