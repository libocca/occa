#include "statement.hpp"

namespace occa {
  namespace lang {
    statement::statement(context_t &context_) :
      up(NULL),
      context(context_),
      scope(context_) {
      addRef();
    }

    statement::~statement() {}

    std::string statement::toString() const {
      std::stringstream ss;
      printer pout(ss);
      print(pout);
      return ss.str();
    }

    statement::operator std::string() const {
      return toString();
    }

    bool statement::hasScope() const {
      return false;
    }

    emptyStatement statement::newEmptyStatement() {
      return emptyStatement(context);
    }

    directiveStatement statement::newDirectiveStatement(macro_t &macro) {
      return directiveStatement(context, macro);
    }

    blockStatement statement::newBlockStatement() {
      return blockStatement(context);
    }

    typeDeclStatement statement::newTypeDeclarationStatement(declarationType &declType) {
      return typeDeclStatement(context, declType);
    }

    classAccessStatement statement::newClassAccessStatement(const int access) {
      return classAccessStatement(context, access);
    }

    expressionStatement statement::newExpressionStatement(exprNode &expression) {
      return expressionStatement(context, expression);
    }

    declarationStatement statement::newDeclarationStatement() {
      return declarationStatement(context);
    }

    gotoStatement statement::  newGotoStatement(const std::string &name) {
      return gotoStatement(context, name);
    }

    gotoLabelStatement statement::newGotoLabelStatement(const std::string &name) {
      return gotoLabelStatement(context, name);
    }

    namespaceStatement statement::newNamespaceStatement(const std::string &name) {
      return namespaceStatement(context, name);
    }

    whileStatement statement::newWhileStatement(statement &check) {
      return whileStatement(context, check, false);
    }

    whileStatement statement::newDoWhileStatement(statement &check) {
      return whileStatement(context, check, true);
    }

    forStatement statement::newForStatement(statement &init,
                                            statement &check,
                                            statement &update) {
      return forStatement(context, init, check, update);
    }

    switchStatement statement::newSwitchStatement(statement &value) {
      return switchStatement(context, value);
    }

    caseStatement statement::newCaseStatement(statement &value) {
      return caseStatement(context, value);
    }

    returnStatement statement::newReturnStatement(statement &value) {
      return returnStatement(context, value);
    }

    void statement::print() const {
      std::cout << toString();
    }

    //---[ Empty ]------------------------
    emptyStatement::emptyStatement(context_t &context_) :
      statement(context_) {}

    statement& emptyStatement::clone() const {
      return *(new emptyStatement(context));
    }

    int emptyStatement::type() const {
      return statementType::empty;
    }

    void emptyStatement::print(printer &pout) const {}
    //====================================

    //---[ Directive ]--------------------
    directiveStatement::directiveStatement(context_t &context_,
                                           macro_t &macro_) :
      statement(context_),
      macro(macro_) {}

    statement& directiveStatement::clone() const {
      return *(new directiveStatement(context, macro));
    }

    int directiveStatement::type() const {
      return statementType::directive;
    }

    void directiveStatement::print(printer &pout) const {
      pout << macro.toString() << '\n';
    }
    //====================================

    //---[ Block ]------------------------
    blockStatement::blockStatement(context_t &context_) :
      statement(context_) {}

    statement& blockStatement::clone() const {
      blockStatement &s = *(new blockStatement(context));
      const int childCount = (int) children.size();
      for (int i = 0; i < childCount; ++i) {
        s.addChild(children[i]->clone());
      }
      return s;
    }

    int blockStatement::type() const {
      return statementType::block;
    }

    bool blockStatement::hasScope() const {
      return true;
    }

    void blockStatement::addChild(statement &child) {
      children.push_back(&child);
      child.up = this;
    }

    void blockStatement::clearChildren() {
      const int count = (int) children.size();
      for (int i = 0; i < count; ++i) {
        if (!children[i]->removeRef()) {
          delete children[i];
        }
      }
      children.clear();
    }

    void blockStatement::print(printer &pout) const {
      // Don't print { } for root statement
      if (up) {
        if (pout.isInlined()) {
          pout << ' ';
        } else {
          pout.printIndentation();
        }
        pout << '{';
        if (children.size()) {
          pout << '\n';
          pout.addIndentation();
        } else {
          pout << ' ';
        }
      }

      printChildren(pout);

      if (up) {
        pout.removeIndentation();
        pout.printIndentation();
        pout << "}\n";
      }
    }

    void blockStatement::printChildren(printer &pout) const {
      const int count = (int) children.size();
      for (int i = 0; i < count; ++i) {
        children[i]->print(pout);
      }
    }
    //====================================

    //---[ Type ]-------------------------
    typeDeclStatement::typeDeclStatement(context_t &context_,
                                         declarationType &declType_) :
      statement(context_),
      declType(declType_) {}


    statement& typeDeclStatement::clone() const {
      return *(new typeDeclStatement(context, declType));
    }

    int typeDeclStatement::type() const {
      return statementType::typeDecl;
    }

    bool typeDeclStatement::hasScope() const {
      return (dynamic_cast<classType*>(&declType)
              || dynamic_cast<functionType*>(&declType));
    }

    void typeDeclStatement::print(printer &pout) const {
      declType.printDeclaration(pout);
    }

    classAccessStatement::classAccessStatement(context_t &context_,
                                               const int access_) :
      statement(context_),
      access(access_) {}

    statement& classAccessStatement::clone() const {
      return *(new classAccessStatement(context, access));
    }

    int classAccessStatement::type() const {
      return statementType::classAccess;
    }

    void classAccessStatement::print(printer &pout) const {
      pout.removeIndentation();

      pout.printIndentation();
      switch (access) {
      case classAccess::private_  : pout << "private:\n"  ; break;
      case classAccess::protected_: pout << "protected:\n"; break;
      case classAccess::public_   : pout << "public:\n"   ; break;
      }

      pout.addIndentation();
    }
    //====================================

    //---[ Expression ]-------------------
    expressionStatement::expressionStatement(context_t &context_,
                                             exprNode &expression_) :
      statement(context_),
      expression(expression_) {}

    statement& expressionStatement::clone() const {
      return *(new expressionStatement(context, expression));
    }

    int expressionStatement::type() const {
      return statementType::expression;
    }

    void expressionStatement::print(printer &pout) const {
    }

    declarationStatement::declarationStatement(context_t &context_) :
      statement(context_) {}

    statement& declarationStatement::clone() const {
      return *(new declarationStatement(context));
    }

    int declarationStatement::type() const {
      return statementType::declaration;
    }

    void declarationStatement::print(printer &pout) const {
    }
    //====================================

    //---[ Goto ]-------------------------
    gotoStatement::gotoStatement(context_t &context_,
                                 const std::string &name_) :
      statement(context_),
      name(name_) {}

    statement& gotoStatement::clone() const {
      return *(new gotoStatement(context, name));
    }

    int gotoStatement::type() const {
      return statementType::goto_;
    }

    void gotoStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "goto " << name << ";\n";
    }

    gotoLabelStatement::gotoLabelStatement(context_t &context_,
                                           const std::string &name_) :
      statement(context_),
      name(name_) {}

    statement& gotoLabelStatement::clone() const {
      return *(new gotoLabelStatement(context, name));
    }

    int gotoLabelStatement::type() const {
      return statementType::gotoLabel;
    }

    void gotoLabelStatement::print(printer &pout) const {
      pout << name << ":\n";
    }
    //====================================

    //---[ Namespace ]--------------------
    namespaceStatement::namespaceStatement(context_t &context_,
                                           const std::string &name_) :
      blockStatement(context_),
      name(name_) {}

    statement& namespaceStatement::clone() const {
      return *(new namespaceStatement(context, name));
    }

    int namespaceStatement::type() const {
      return statementType::namespace_;
    }

    bool namespaceStatement::hasScope() const {
      return true;
    }

    void namespaceStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "namespace " << name << " {\n";

      pout.pushInlined(false);
      pout.addIndentation();
      printChildren(pout);
      pout.removeIndentation();
      pout.popInlined();

      pout.printIndentation();
      pout << "}\n";
    }
    //====================================

    //---[ While ]------------------------
    whileStatement::whileStatement(context_t &context_,
                                   statement &check_,
                                   const bool isDoWhile_) :
      blockStatement(context_),
      check(check_),
      isDoWhile(isDoWhile_) {}

    statement& whileStatement::clone() const {
      return *(new whileStatement(context, check.clone(), isDoWhile));
    }

    int whileStatement::type() const {
      return statementType::while_;
    }

    bool whileStatement::hasScope() const {
      return true;
    }

    void whileStatement::print(printer &pout) const {
      pout.printStartIndentation();
      if (isDoWhile) {
        pout << "while (";
        pout.pushInlined(true);
        check.print(pout);
        pout.popInlined();
        pout << ')';
      } else {
        pout << "do";
      }
      pout << " {\n";

      pout.pushInlined(false);
      pout.addIndentation();
      printChildren(pout);
      pout.removeIndentation();
      pout.popInlined();

      pout.printIndentation();
      pout << '}';
      if (isDoWhile) {
        pout << " while (";
        pout.pushInlined(true);
        check.print(pout);
        pout.popInlined();
        pout << ')';
      }
      pout.printEndNewline();
    }
    //====================================

    //---[ For ]--------------------------
    forStatement::forStatement(context_t &context_,
                               statement &init_,
                               statement &check_,
                               statement &update_) :
      blockStatement(context_),
      init(init_),
      check(check_),
      update(update_) {}

    statement& forStatement::clone() const {
      return *(new forStatement(context,
                                init.clone(),
                                check.clone(),
                                update.clone()));
    }

    int forStatement::type() const {
      return statementType::for_;
    }

    bool forStatement::hasScope() const {
      return true;
    }

    void forStatement::print(printer &pout) const {
      pout.printStartIndentation();
      pout << "for (";
      pout.pushInlined(true);
      init.print(pout);
      check.print(pout);
      update.print(pout);
      pout.popInlined();
      pout << ") {\n";

      pout.pushInlined(false);
      pout.addIndentation();
      printChildren(pout);
      pout.removeIndentation();
      pout.popInlined();

      pout.printIndentation();
      pout << '}';
      pout.printEndNewline();
    }
    //====================================

    //---[ Switch ]-----------------------
    switchStatement::switchStatement(context_t &context_,
                                     statement &value_) :
      blockStatement(context_),
      value(value_) {}

    statement& switchStatement::clone() const {
      return *(new switchStatement(context, value.clone()));
    }

    int switchStatement::type() const {
      return statementType::switch_;
    }

    bool switchStatement::hasScope() const {
      return true;
    }

    void switchStatement::print(printer &pout) const {
      pout.printStartIndentation();
      pout << "switch (";
      pout.pushInlined(true);
      value.print(pout);
      pout.popInlined();
      pout << ") {\n";

      pout.pushInlined(false);
      pout.addIndentation();
      printChildren(pout);
      pout.removeIndentation();
      pout.popInlined();

      pout.printIndentation();
      pout << '}';
      pout.printEndNewline();
    }
    //====================================

    //---[ Case ]-------------------------
    caseStatement::caseStatement(context_t &context_,
                                 statement &value_) :
      statement(context_),
      value(value_) {}

    statement& caseStatement::clone() const {
      return *(new caseStatement(context, value.clone()));
    }

    int caseStatement::type() const {
      return statementType::case_;
    }

    void caseStatement::print(printer &pout) const {
      pout.removeIndentation();

      pout.printIndentation();
      pout << "case ";
      pout.pushInlined(true);
      value.print(pout);
      pout.popInlined();
      pout << ":\n";

      pout.addIndentation();
    }
    //====================================

    //---[ Exit ]-------------------------
    continueStatement::continueStatement(context_t &context_) :
      statement(context_) {}

    statement& continueStatement::clone() const {
      return *(new continueStatement(context));
    }

    int continueStatement::type() const {
      return statementType::continue_;
    }

    void continueStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "continue;\n";
    }

    breakStatement::breakStatement(context_t &context_) :
      statement(context_) {}

    statement& breakStatement::clone() const {
      return *(new breakStatement(context));
    }

    int breakStatement::type() const {
      return statementType::break_;
    }

    void breakStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "break;\n";
    }

    returnStatement::returnStatement(context_t &context_,
                                     statement &value_) :
      statement(context_),
      value(value_) {}

    statement& returnStatement::clone() const {
      return *(new returnStatement(context, value.clone()));
    }

    int returnStatement::type() const {
      return statementType::return_;
    }

    void returnStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "return";
      if (value.type() != statementType::empty) {
        pout << ' ';
        pout.pushInlined(true);
        value.print(pout);
        pout.popInlined();
      }
      pout << ";\n";
    }
    //====================================
  }
}
