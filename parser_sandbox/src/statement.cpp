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
#include "type.hpp"

namespace occa {
  namespace lang {
    statement_t::statement_t(scope_t &scope_) :
      up(NULL),
      scope(scope_),
      attributes() {}

    statement_t::~statement_t() {}

    std::string statement_t::toString() const {
      std::stringstream ss;
      printer pout(ss);
      print(pout);
      return ss.str();
    }

    statement_t::operator std::string() const {
      return toString();
    }

    bool statement_t::hasScope() const {
      return false;
    }

    void statement_t::addAttribute(const attribute_t &attribute) {
      // TODO: Warning if attribute already exists
      // Override last attribute by default
      attributes.push_back(attribute);
    }

    emptyStatement statement_t::newEmptyStatement() {
      return emptyStatement(scope);
    }

    pragmaStatement statement_t::newPragmaStatement(const std::string &line) {
      return pragmaStatement(scope, line);
    }

    blockStatement statement_t::newBlockStatement() {
      return blockStatement(scope);
    }

    typeDeclStatement statement_t::newTypeDeclarationStatement(declarationType &declType) {
      return typeDeclStatement(scope, declType);
    }

    classAccessStatement statement_t::newClassAccessStatement(const int access) {
      return classAccessStatement(scope, access);
    }

    expressionStatement statement_t::newExpressionStatement(exprNode &expression) {
      return expressionStatement(scope, expression);
    }

    declarationStatement statement_t::newDeclarationStatement() {
      return declarationStatement(scope);
    }

    gotoStatement statement_t::  newGotoStatement(const std::string &name) {
      return gotoStatement(scope, name);
    }

    gotoLabelStatement statement_t::newGotoLabelStatement(const std::string &name) {
      return gotoLabelStatement(scope, name);
    }

    namespaceStatement statement_t::newNamespaceStatement(const std::string &name) {
      return namespaceStatement(scope, name);
    }

    whileStatement statement_t::newWhileStatement(statement_t &check) {
      return whileStatement(scope, check, false);
    }

    whileStatement statement_t::newDoWhileStatement(statement_t &check) {
      return whileStatement(scope, check, true);
    }

    forStatement statement_t::newForStatement(statement_t &init,
                                              statement_t &check,
                                              statement_t &update) {
      return forStatement(scope, init, check, update);
    }

    switchStatement statement_t::newSwitchStatement(statement_t &value) {
      return switchStatement(scope, value);
    }

    caseStatement statement_t::newCaseStatement(statement_t &value) {
      return caseStatement(scope, value);
    }

    returnStatement statement_t::newReturnStatement(statement_t &value) {
      return returnStatement(scope, value);
    }

    void statement_t::print() const {
      std::cout << toString();
    }

    //---[ Empty ]------------------------
    emptyStatement::emptyStatement(scope_t &scope_) :
      statement_t(scope_) {}

    statement_t& emptyStatement::clone() const {
      emptyStatement &s = *(new emptyStatement(scope));
      s.attributes = attributes;
      return s;
    }

    int emptyStatement::type() const {
      return statementType::empty;
    }

    void emptyStatement::print(printer &pout) const {}
    //====================================

    //---[ Pragma ]--------------------
    pragmaStatement::pragmaStatement(scope_t &scope_,
                                     const std::string &line_) :
      statement_t(scope_),
      line(line_) {}

    statement_t& pragmaStatement::clone() const {
      pragmaStatement &s = *(new pragmaStatement(scope, line));
      s.attributes = attributes;
      return s;
    }

    int pragmaStatement::type() const {
      return statementType::pragma;
    }

    void pragmaStatement::print(printer &pout) const {
      pout << "#pragma " << line << '\n';
    }
    //====================================

    //---[ Block ]------------------------
    blockStatement::blockStatement(scope_t &scope_) :
      statement_t(scope_) {}

    statement_t& blockStatement::clone() const {
      blockStatement &s = *(new blockStatement(scope));
      s.attributes = attributes;
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

    void blockStatement::addChild(statement_t &child) {
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
    typeDeclStatement::typeDeclStatement(scope_t &scope_,
                                         declarationType &declType_) :
      statement_t(scope_),
      declType(declType_) {}


    statement_t& typeDeclStatement::clone() const {
      typeDeclStatement &s = *(new typeDeclStatement(scope, declType));
      s.attributes = attributes;
      return s;
    }

    int typeDeclStatement::type() const {
      return statementType::typeDecl;
    }

    bool typeDeclStatement::hasScope() const {
      return (declType.is<structureType>() ||
              declType.is<functionType>());
    }

    void typeDeclStatement::print(printer &pout) const {
      declType.printDeclaration(pout);
    }

    classAccessStatement::classAccessStatement(scope_t &scope_,
                                               const int access_) :
      statement_t(scope_),
      access(access_) {}

    statement_t& classAccessStatement::clone() const {
      classAccessStatement &s = *(new classAccessStatement(scope, access));
      s.attributes = attributes;
      return s;
    }

    int classAccessStatement::type() const {
      return statementType::classAccess;
    }

    void classAccessStatement::print(printer &pout) const {
      pout.removeIndentation();

      pout.printIndentation();
      if (access & classAccess::public_) {
        pout << "public:\n";
      }
      else if (access & classAccess::private_) {
        pout << "private:\n";
      }
      else if (access & classAccess::protected_) {
        pout << "protected:\n";
      }

      pout.addIndentation();
    }
    //====================================

    //---[ Expression ]-------------------
    expressionStatement::expressionStatement(scope_t &scope_,
                                             exprNode &expression_) :
      statement_t(scope_),
      expression(expression_) {}

    statement_t& expressionStatement::clone() const {
      expressionStatement &s = *(new expressionStatement(scope, expression));
      s.attributes = attributes;
      return s;
    }

    int expressionStatement::type() const {
      return statementType::expression;
    }

    void expressionStatement::print(printer &pout) const {
    }

    declarationStatement::declarationStatement(scope_t &scope_) :
      statement_t(scope_) {}

    statement_t& declarationStatement::clone() const {
      declarationStatement &s = *(new declarationStatement(scope));
      s.attributes = attributes;
      return s;
    }

    int declarationStatement::type() const {
      return statementType::declaration;
    }

    void declarationStatement::print(printer &pout) const {
    }
    //====================================

    //---[ Goto ]-------------------------
    gotoStatement::gotoStatement(scope_t &scope_,
                                 const std::string &name_) :
      statement_t(scope_),
      name(name_) {}

    statement_t& gotoStatement::clone() const {
      gotoStatement &s = *(new gotoStatement(scope, name));
      s.attributes = attributes;
      return s;
    }

    int gotoStatement::type() const {
      return statementType::goto_;
    }

    void gotoStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "goto " << name << ";\n";
    }

    gotoLabelStatement::gotoLabelStatement(scope_t &scope_,
                                           const std::string &name_) :
      statement_t(scope_),
      name(name_) {}

    statement_t& gotoLabelStatement::clone() const {
      gotoLabelStatement &s = *(new gotoLabelStatement(scope, name));
      s.attributes = attributes;
      return s;
    }

    int gotoLabelStatement::type() const {
      return statementType::gotoLabel;
    }

    void gotoLabelStatement::print(printer &pout) const {
      pout << name << ":\n";
    }
    //====================================

    //---[ Namespace ]--------------------
    namespaceStatement::namespaceStatement(scope_t &scope_,
                                           const std::string &name_) :
      blockStatement(scope_),
      name(name_) {}

    statement_t& namespaceStatement::clone() const {
      namespaceStatement &s = *(new namespaceStatement(scope, name));
      s.attributes = attributes;
      return s;
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
    whileStatement::whileStatement(scope_t &scope_,
                                   statement_t &check_,
                                   const bool isDoWhile_) :
      blockStatement(scope_),
      check(check_),
      isDoWhile(isDoWhile_) {}

    statement_t& whileStatement::clone() const {
      whileStatement &s = *(new whileStatement(scope, check.clone(), isDoWhile));
      s.attributes = attributes;
      return s;
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
    forStatement::forStatement(scope_t &scope_,
                               statement_t &init_,
                               statement_t &check_,
                               statement_t &update_) :
      blockStatement(scope_),
      init(init_),
      check(check_),
      update(update_) {}

    statement_t& forStatement::clone() const {
      forStatement &s = *(new forStatement(scope,
                                           init.clone(),
                                           check.clone(),
                                           update.clone()));
      s.attributes = attributes;
      return s;
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
    switchStatement::switchStatement(scope_t &scope_,
                                     statement_t &value_) :
      blockStatement(scope_),
      value(value_) {}

    statement_t& switchStatement::clone() const {
      switchStatement &s = *(new switchStatement(scope, value.clone()));
      s.attributes = attributes;
      return s;
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
    caseStatement::caseStatement(scope_t &scope_,
                                 statement_t &value_) :
      statement_t(scope_),
      value(value_) {}

    statement_t& caseStatement::clone() const {
      caseStatement &s = *(new caseStatement(scope, value.clone()));
      s.attributes = attributes;
      return s;
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
    continueStatement::continueStatement(scope_t &scope_) :
      statement_t(scope_) {}

    statement_t& continueStatement::clone() const {
      continueStatement &s = *(new continueStatement(scope));
      s.attributes = attributes;
      return s;
    }

    int continueStatement::type() const {
      return statementType::continue_;
    }

    void continueStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "continue;\n";
    }

    breakStatement::breakStatement(scope_t &scope_) :
      statement_t(scope_) {}

    statement_t& breakStatement::clone() const {
      breakStatement &s = *(new breakStatement(scope));
      s.attributes = attributes;
      return s;
    }

    int breakStatement::type() const {
      return statementType::break_;
    }

    void breakStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "break;\n";
    }

    returnStatement::returnStatement(scope_t &scope_,
                                     statement_t &value_) :
      statement_t(scope_),
      value(value_) {}

    statement_t& returnStatement::clone() const {
      returnStatement &s = *(new returnStatement(scope, value.clone()));
      s.attributes = attributes;
      return s;
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
