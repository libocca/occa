/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
#include "statement.hpp"

namespace occa {
  namespace lang {
    statement::statement(context &ctx_) :
      up(NULL),
      ctx(ctx_),
      scope(ctx_) {}

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
      return emptyStatement(ctx);
    }

    directiveStatement statement::newDirectiveStatement(macro_t &macro) {
      return directiveStatement(ctx, macro);
    }

    blockStatement statement::newBlockStatement() {
      return blockStatement(ctx);
    }

    typeDeclStatement statement::newTypeDeclarationStatement(declarationType &declType) {
      return typeDeclStatement(ctx, declType);
    }

    classAccessStatement statement::newClassAccessStatement(const int access) {
      return classAccessStatement(ctx, access);
    }

    expressionStatement statement::newExpressionStatement(exprNode &expression) {
      return expressionStatement(ctx, expression);
    }

    declarationStatement statement::newDeclarationStatement() {
      return declarationStatement(ctx);
    }

    gotoStatement statement::  newGotoStatement(const std::string &name) {
      return gotoStatement(ctx, name);
    }

    gotoLabelStatement statement::newGotoLabelStatement(const std::string &name) {
      return gotoLabelStatement(ctx, name);
    }

    namespaceStatement statement::newNamespaceStatement(const std::string &name) {
      return namespaceStatement(ctx, name);
    }

    whileStatement statement::newWhileStatement(statement &check) {
      return whileStatement(ctx, check, false);
    }

    whileStatement statement::newDoWhileStatement(statement &check) {
      return whileStatement(ctx, check, true);
    }

    forStatement statement::newForStatement(statement &init,
                                            statement &check,
                                            statement &update) {
      return forStatement(ctx, init, check, update);
    }

    switchStatement statement::newSwitchStatement(statement &value) {
      return switchStatement(ctx, value);
    }

    caseStatement statement::newCaseStatement(statement &value) {
      return caseStatement(ctx, value);
    }

    returnStatement statement::newReturnStatement(statement &value) {
      return returnStatement(ctx, value);
    }

    void statement::print() const {
      std::cout << toString();
    }

    //---[ Empty ]------------------------
    emptyStatement::emptyStatement(context &ctx_) :
      statement(ctx_) {}

    statement& emptyStatement::clone() const {
      return *(new emptyStatement(ctx));
    }

    int emptyStatement::type() const {
      return statementType::empty;
    }

    void emptyStatement::print(printer &pout) const {}
    //====================================

    //---[ Directive ]--------------------
    directiveStatement::directiveStatement(context &ctx_,
                                           macro_t &macro_) :
      statement(ctx_),
      macro(macro_) {}

    statement& directiveStatement::clone() const {
      return *(new directiveStatement(ctx, macro));
    }

    int directiveStatement::type() const {
      return statementType::directive;
    }

    void directiveStatement::print(printer &pout) const {
      pout << macro.toString() << '\n';
    }
    //====================================

    //---[ Block ]------------------------
    blockStatement::blockStatement(context &ctx_) :
      statement(ctx_) {}

    statement& blockStatement::clone() const {
      blockStatement &s = *(new blockStatement(ctx));
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
        delete children[i];
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
    typeDeclStatement::typeDeclStatement(context &ctx_,
                                         declarationType &declType_) :
      statement(ctx_),
      declType(declType_) {}


    statement& typeDeclStatement::clone() const {
      return *(new typeDeclStatement(ctx, declType));
    }

    int typeDeclStatement::type() const {
      return statementType::typeDecl;
    }

    bool typeDeclStatement::hasScope() const {
      return (declType.is<classType>() ||
              declType.is<functionType>());
    }

    void typeDeclStatement::print(printer &pout) const {
      declType.printDeclaration(pout);
    }

    classAccessStatement::classAccessStatement(context &ctx_,
                                               const int access_) :
      statement(ctx_),
      access(access_) {}

    statement& classAccessStatement::clone() const {
      return *(new classAccessStatement(ctx, access));
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
    expressionStatement::expressionStatement(context &ctx_,
                                             exprNode &expression_) :
      statement(ctx_),
      expression(expression_) {}

    statement& expressionStatement::clone() const {
      return *(new expressionStatement(ctx, expression));
    }

    int expressionStatement::type() const {
      return statementType::expression;
    }

    void expressionStatement::print(printer &pout) const {
    }

    declarationStatement::declarationStatement(context &ctx_) :
      statement(ctx_) {}

    statement& declarationStatement::clone() const {
      return *(new declarationStatement(ctx));
    }

    int declarationStatement::type() const {
      return statementType::declaration;
    }

    void declarationStatement::print(printer &pout) const {
    }
    //====================================

    //---[ Goto ]-------------------------
    gotoStatement::gotoStatement(context &ctx_,
                                 const std::string &name_) :
      statement(ctx_),
      name(name_) {}

    statement& gotoStatement::clone() const {
      return *(new gotoStatement(ctx, name));
    }

    int gotoStatement::type() const {
      return statementType::goto_;
    }

    void gotoStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "goto " << name << ";\n";
    }

    gotoLabelStatement::gotoLabelStatement(context &ctx_,
                                           const std::string &name_) :
      statement(ctx_),
      name(name_) {}

    statement& gotoLabelStatement::clone() const {
      return *(new gotoLabelStatement(ctx, name));
    }

    int gotoLabelStatement::type() const {
      return statementType::gotoLabel;
    }

    void gotoLabelStatement::print(printer &pout) const {
      pout << name << ":\n";
    }
    //====================================

    //---[ Namespace ]--------------------
    namespaceStatement::namespaceStatement(context &ctx_,
                                           const std::string &name_) :
      blockStatement(ctx_),
      name(name_) {}

    statement& namespaceStatement::clone() const {
      return *(new namespaceStatement(ctx, name));
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
    whileStatement::whileStatement(context &ctx_,
                                   statement &check_,
                                   const bool isDoWhile_) :
      blockStatement(ctx_),
      check(check_),
      isDoWhile(isDoWhile_) {}

    statement& whileStatement::clone() const {
      return *(new whileStatement(ctx, check.clone(), isDoWhile));
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
    forStatement::forStatement(context &ctx_,
                               statement &init_,
                               statement &check_,
                               statement &update_) :
      blockStatement(ctx_),
      init(init_),
      check(check_),
      update(update_) {}

    statement& forStatement::clone() const {
      return *(new forStatement(ctx,
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
    switchStatement::switchStatement(context &ctx_,
                                     statement &value_) :
      blockStatement(ctx_),
      value(value_) {}

    statement& switchStatement::clone() const {
      return *(new switchStatement(ctx, value.clone()));
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
    caseStatement::caseStatement(context &ctx_,
                                 statement &value_) :
      statement(ctx_),
      value(value_) {}

    statement& caseStatement::clone() const {
      return *(new caseStatement(ctx, value.clone()));
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
    continueStatement::continueStatement(context &ctx_) :
      statement(ctx_) {}

    statement& continueStatement::clone() const {
      return *(new continueStatement(ctx));
    }

    int continueStatement::type() const {
      return statementType::continue_;
    }

    void continueStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "continue;\n";
    }

    breakStatement::breakStatement(context &ctx_) :
      statement(ctx_) {}

    statement& breakStatement::clone() const {
      return *(new breakStatement(ctx));
    }

    int breakStatement::type() const {
      return statementType::break_;
    }

    void breakStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "break;\n";
    }

    returnStatement::returnStatement(context &ctx_,
                                     statement &value_) :
      statement(ctx_),
      value(value_) {}

    statement& returnStatement::clone() const {
      return *(new returnStatement(ctx, value.clone()));
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
