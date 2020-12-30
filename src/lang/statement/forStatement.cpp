#include <occa/internal/lang/statement/forStatement.hpp>

namespace occa {
  namespace lang {
    forStatement::forStatement(blockStatement *up_,
                               token_t *source_) :
        blockStatement(up_, source_),
        init(NULL),
        check(NULL),
        update(NULL) {}

    forStatement::forStatement(blockStatement *up_,
                               const forStatement &other) :
        blockStatement(up_, other.source),
        init(statement_t::clone(this, other.init)),
        check(statement_t::clone(this, other.check)),
        update(statement_t::clone(this, other.update)) {

      copyFrom(other);
    }

    forStatement::~forStatement() {
      delete init;
      delete check;
      delete update;
    }

    void forStatement::setLoopStatements(statement_t *init_,
                                         statement_t *check_,
                                         statement_t *update_) {
      init   = init_;
      check  = check_;
      update = update_;
      if (init) {
        init->up = this;
      }
      if (check) {
        check->up = this;
      }
      if (update) {
        update->up = this;
      }
    }

    statement_t& forStatement::clone_(blockStatement *up_) const {
      return *(new forStatement(up_, *this));
    }

    int forStatement::type() const {
      return statementType::for_;
    }

    std::string forStatement::statementName() const {
      return "for";
    }

    statementArray forStatement::getInnerStatements() {
      statementArray arr;

      if (init) {
        arr.push(init);
      }
      if (check) {
        arr.push(check);
      }
      if (update) {
        arr.push(update);
      }

      return arr;
    }

    void forStatement::print(printer &pout) const {
      pout.printStartIndentation();

      pout << "for (";
      pout.pushInlined(true);

      // When developing or making code transformations, we might
      // not always have all 3 set
      if (init) {
        pout << *init;
      }
      if (check) {
        pout << *check;
      }
      if (update) {
        pout << *update;
      }

      pout << ')';

      blockStatement::print(pout);
      pout.popInlined();
    }
  }
}
