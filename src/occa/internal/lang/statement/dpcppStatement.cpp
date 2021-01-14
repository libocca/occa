#include <occa/internal/lang/statement/dpcppStatement.hpp>

namespace occa {
  namespace lang {
    dpcppStatement::dpcppStatement(blockStatement *up_,
                                   token_t *source_) :
      blockStatement(up_, source_) {}

    dpcppStatement::dpcppStatement(blockStatement *up_,
                                   const dpcppStatement &other) :
      blockStatement(up_, other) {
      copyFrom(other);
    }

    dpcppStatement::~dpcppStatement() {
      clear();
    }


    statement_t& dpcppStatement::clone_(dpcppStatement *up_) const {
      return *(new dpcppStatement(up_, *this));
    }

    int dpcppStatement::type() const {
      return statementType::block;
    }

    std::string dpcppStatement::statementName() const {
      return "block";
    }

     bool dpcppStatement::hasInScope(const std::string &name) {
      if (scope.has(name)) {
        return true;
      }
      return (up
              ? up->hasInScope(name)
              : false);
    }

    keyword_t& dpcppStatement::getScopeKeyword(const std::string &name) {
      keyword_t &keyword = scope.get(name);
      if ((keyword.type() == keywordType::none)
          && up) {
        return up->getScopeKeyword(name);
      }
      return keyword;
    }



    void dpcppStatement::print(printer &pout) const {
      bool hasChildren = children.size();
      if (!hasChildren) {
        if (up) {
          pout.printStartIndentation();
          pout << "{}\n";
        }
        return;
      }

      // Don't print { } for root statement
      if (up) {
        pout.printStartIndentation();
        pout.pushInlined(false);
        pout << "q->submit([&](sycl::handler &h){\n";
        pout.addIndentation();
        pout << "h.parallel_for(*ndrange, [=] (auto ii){\n";
        pout.addIndentation();
        pout << "int i = ii.get_global_linear_id();\n";
      }

      printChildren(pout);

      if (up) {
        pout.removeIndentation();
        pout.popInlined();
        pout.printNewline();
        pout.printIndentation();
        pout << "});\n";
        pout.removeIndentation();
        pout.popInlined();
        pout << "});\n";
      }
    }

    void dpcppStatement::printChildren(printer &pout) const {
      const int count = (int) children.size();
      for (int i = 0; i < count; ++i) {
        pout << *(children[i]);
      }
    }
  }
}
