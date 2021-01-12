#include <occa/internal/lang/statement/ifStatement.hpp>
#include <occa/internal/lang/statement/elifStatement.hpp>
#include <occa/internal/lang/statement/elseStatement.hpp>

namespace occa {
  namespace lang {
    ifStatement::ifStatement(blockStatement *up_,
                             token_t *source_) :
      blockStatement(up_, source_),
      condition(NULL),
      elseSmnt(NULL) {}

    ifStatement::ifStatement(blockStatement *up_,
                             const ifStatement &other) :
      blockStatement(up_, other.source),
      condition(&(other.condition->clone(this))),
      elseSmnt(NULL) {

      copyFrom(other);

      const int elifCount = (int) other.elifSmnts.size();
      for (int i = 0; i < elifCount; ++i) {
        elifStatement &elifSmnt = (other.elifSmnts[i]
                                   ->clone(this)
                                   .to<elifStatement>());
        elifSmnts.push_back(&elifSmnt);
      }

      if (other.elseSmnt) {
        elseSmnt = &((elseStatement&) other.elseSmnt->clone(this));
      }
    }

    ifStatement::~ifStatement() {
      delete condition;
      delete elseSmnt;

      const int elifCount = (int) elifSmnts.size();
      for (int i = 0; i < elifCount; ++i) {
        delete elifSmnts[i];
      }
    }

    void ifStatement::setCondition(statement_t *condition_) {
      condition = condition_;
    }

    void ifStatement::addElif(elifStatement &elifSmnt) {
      elifSmnts.push_back(&elifSmnt);
    }

    void ifStatement::addElse(elseStatement &elseSmnt_) {
      delete elseSmnt;
      elseSmnt = &elseSmnt_;
    }

    statement_t& ifStatement::clone_(blockStatement *up_) const {
      return *(new ifStatement(up_, *this));
    }

    int ifStatement::type() const {
      return statementType::if_;
    }

    std::string ifStatement::statementName() const {
      return "if";
    }

    statementArray ifStatement::getInnerStatements() {
      statementArray arr;

      if (condition) {
        arr.push(condition);
      }
      for (statement_t *elifSmnt : elifSmnts) {
        arr.push(elifSmnt);
      }
      if (elseSmnt) {
        arr.push(elseSmnt);
      }

      return arr;
    }

    void ifStatement::print(printer &pout) const {
      pout.printStartIndentation();
      pout << "if (";
      pout.pushInlined(true);
      condition->print(pout);
      pout << ')';

      blockStatement::print(pout);
      pout.popInlined();

      const int elifCount = (int) elifSmnts.size();
      for (int i = 0; i < elifCount; ++i) {
        pout << *(elifSmnts[i]);
      }

      if (elseSmnt) {
        pout << (*elseSmnt);
      }
    }
  }
}
