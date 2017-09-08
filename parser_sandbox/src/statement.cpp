#if 0

#include "statement.hpp"

namespace occa {
  statementPrintInfo_t::statementPrintInfo_t() :
    s(),
    indent(),
    inlined(false) {}

  void statementPrintInfo_t::addIndentation() {
    indent += "  ";
  }

  void statementPrintInfo_t::removeIndentation() {
    const int chars = (int) indent.size();
    if (chars >= 2) {
      indent.resize(chars - 2);
    }
  }

  statement_t::statement(parser_t &parser_) :
    parser(parser_),
    scope(),
    root(),
    up(NULL) {

    addRef();
  }

  ~statement_t::statement_t() {}

  std::string statement_t::toString() const {
    std::stringstream ss;
    std::string indent;
    print(pi);
    return ss.str();
  }

  statement_t::operator std::string() const {
    return toString();
  }

  void statement_t::print() const {
    std::cout << toString();
  }

  //---[ Directive ]--------------------
  statement_t& directiveStatement_t::clone() const {
    return *this;
  }

  void directiveStatement_t::print(statementPrintInfo_t &pi) const {
    pi.s += directive;
    pi.s += '\n';
  }
  //====================================

  //---[ Type ]-------------------------
  statement_t& typeStatement_t::clone() const {
    return *this;
  }

  void typeStatement_t::print(statementPrintInfo_t &pi) const {
  }
  //====================================

  //---[ Declaration ]------------------
  statement_t& declarationStatement_t::clone() const {
    return *this;
  }

  void declarationStatement_t::print(statementPrintInfo_t &pi) const {
  }
  //====================================

  //---[ Expression ]-------------------
  statement_t& expressionStatement_t::clone() const {
    return *this;
  }

  void expressionStatement_t::print(statementPrintInfo_t &pi) const {
  }
  //====================================

  //---[ Goto ]-------------------------
  statement_t& gotoStatement_t::clone() const {
    return *this;
  }

  void gotoStatement_t::print(statementPrintInfo_t &pi) const {
    pi.s += indent;
    pi.s += "goto ";
    pi.s += name;
    pi.s += '\n';
  }
  //====================================

  //---[ Block ]------------------------
  statement_t& blockStatement_t::clone() const {
    return *this;
  }

  void blockStatement_t::addChild(statement_t *child) {
    children.push_back(child);
    child.up = this;
  }

  void blockStatement_t::clearChildren() {
    const int count = (int) children.size();
    for (int i = 0; i < count; ++i) {
      if (!children[i]->removeRef()) {
        delete children[i];
      }
    }
    children.clear();
  }

  void blockStatement_t::print(statementPrintInfo_t &pi) const {
    // Don't print { } for root statement
    if (up) {
      pi.s += indent;
      pi.s += "{\n";
      pi.addIndentation();
    }

    printChildren(pi);

    if (up) {
      pi.removeIndentation();
      pi.s += indent;
      pi.s += "}\n";
    }
  }

  void blockStatement_t::printChildren(statementPrintInfo_t pi) const {
    const int count = (int) children.size();
    for (int i = 0; i < count; ++i) {
      children[i]->print(pi);
    }
  }
  //====================================

  //---[ Namespace ]--------------------
  statement_t& namespaceStatement_t::clone() const {
    return *this;
  }

  void namespaceStatement_t::print(statementPrintInfo_t &pi) const {
    pi.s += indent;
    pi.s += "namespace ";
    pi.s += name;
    pi.s += " {\n";
    pi.addIndentation();

    printChildren(pi);

    pi.removeIndentation();
    pi.s += indent;
    pi.s += "}\n";
  }
  //====================================

  //---[ While ]------------------------
  statement_t& whileStatement_t::clone() const {
    return *this;
  }

  void whileStatement_t::print(statementPrintInfo_t &pi) const {
    pi.s += indent;
    pi.s += "while(";
    pi.inlined = true;
    check.print(pi);
    pi.inlined = false;
    pi.s += ") {\n";

    pi.addIndentation();
    printChildren(pi);
    pi.removeIndentation();

    pi.s += indent;
    pi.s += "}\n";
  }
  //====================================

  //---[ For ]--------------------------
  statement_t& forStatement_t::clone() const {
    return *this;
  }

  void forStatement_t::print(statementPrintInfo_t &pi) const {
    pi.s += indent;
    pi.s += "for(";
    pi.inlined = true;
    init.print(pi);
    pi.s += ' ';
    check.print(pi);
    pi.s += ' ';
    update.print(pi);
    pi.inlined = false;
    pi.s += ") {\n";

    pi.addIndentation();
    printChildren(pi);
    pi.removeIndentation();

    pi.s += indent;
    pi.s += "}\n";
  }
  //====================================

  //---[ Switch ]-----------------------
  statement_t& switchStatement_t::clone() const {
    return *this;
  }

  void switchStatement_t::print(statementPrintInfo_t &pi) const {
    pi.s += indent;
    pi.s += "switch(";
    pi.inlined = true;
    value.print(pi);
    pi.inlined = false;
    pi.s += ") {\n";

    pi.addIndentation();
    printChildren(pi);
    pi.removeIndentation();

    pi.s += indent;
    pi.s += "}\n";
  }
  //====================================

  //---[ Case ]-------------------------
  statement_t& caseStatement_t::clone() const {
    return *this;
  }

  void caseStatement_t::print(statementPrintInfo_t &pi) const {
    pi.removeIndentation();
    pi.s += indent;
    pi.s += "case ";
    pi.inlined = true;
    value.print(pi);
    pi.inlined = false;
    pi.s += ":\n";
    pi.addIndentation();
  }
  //====================================
}

#endif
