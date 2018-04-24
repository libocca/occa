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
#include "expression.hpp"
#include "statement.hpp"
#include "type.hpp"

namespace occa {
  namespace lang {
    //---[ Pragma ]--------------------
    pragmaStatement::pragmaStatement(pragmaToken &token_) :
      token(token_) {}

    pragmaStatement::~pragmaStatement() {
      delete &token;
    }

    statement_t& pragmaStatement::clone_() const {
      return *(new pragmaStatement(token.clone()->to<pragmaToken>()));
    }

    int pragmaStatement::type() const {
      return statementType::pragma;
    }

    void pragmaStatement::print(printer &pout) const {
      pout << "#pragma " << token.value << '\n';
    }
    //==================================

    //---[ Type ]-----------------------
    classAccessStatement::classAccessStatement(const int access_) :
      statement_t(),
      access(access_) {}

    statement_t& classAccessStatement::clone_() const {
      return *(new classAccessStatement(access));
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
    //==================================

    //---[ Expression ]-----------------
    expressionStatement::expressionStatement(exprNode &root_) :
      statement_t(),
      root(&root_) {}

    expressionStatement::~expressionStatement() {
      delete root;
    }

    statement_t& expressionStatement::clone_() const {
      return *(new expressionStatement(root->clone()));
    }

    int expressionStatement::type() const {
      return statementType::expression;
    }

    void expressionStatement::print(printer &pout) const {
      pout << (*root);
    }

    declarationStatement::declarationStatement() :
      statement_t() {}

    declarationStatement::declarationStatement(const declarationStatement &other) :
      declarations(other.declarations) {}

    statement_t& declarationStatement::clone_() const {
      return *(new declarationStatement(*this));
    }

    int declarationStatement::type() const {
      return statementType::declaration;
    }

    void declarationStatement::print(printer &pout) const {
      const int count = (int) declarations.size();
      if (!count) {
        return;
      }
      pout.printIndentation();
      declarations[0].print(pout);
      for (int i = 1; i < count; ++i) {
        pout << ", ";
        declarations[i].printAsExtra(pout);
      }
      pout << ";\n";
    }
    //==================================

    //---[ Goto ]-----------------------
    gotoStatement::gotoStatement(const std::string &name_) :
      statement_t(),
      name(name_) {}

    statement_t& gotoStatement::clone_() const {
      return *(new gotoStatement(name));
    }

    int gotoStatement::type() const {
      return statementType::goto_;
    }

    void gotoStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "goto " << name << ";\n";
    }

    gotoLabelStatement::gotoLabelStatement(const std::string &name_) :
      statement_t(),
      name(name_) {}

    statement_t& gotoLabelStatement::clone_() const {
      return *(new gotoLabelStatement(name));
    }

    int gotoLabelStatement::type() const {
      return statementType::gotoLabel;
    }

    void gotoLabelStatement::print(printer &pout) const {
      pout << name << ":\n";
    }
    //==================================

    //---[ Namespace ]------------------
    namespaceStatement::namespaceStatement(identifierToken &nameToken_) :
      blockStatement(),
      nameToken(nameToken_) {}

    namespaceStatement::namespaceStatement(const namespaceStatement &other) :
      blockStatement(other),
      nameToken(other.nameToken.clone()->to<identifierToken>()) {}

    namespaceStatement::~namespaceStatement() {
      delete &nameToken;
    }

    statement_t& namespaceStatement::clone_() const {
      return *(new namespaceStatement(*this));
    }

    std::string& namespaceStatement::name() {
      return nameToken.value;
    }

    const std::string& namespaceStatement::name() const {
      return nameToken.value;
    }

    int namespaceStatement::type() const {
      return statementType::namespace_;
    }

    void namespaceStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "namespace " << name();

      blockStatement::print(pout);
    }
    //==================================

    //---[ If ]-------------------------
    ifStatement::ifStatement(statement_t *condition_) :
      condition(condition_),
      elseSmnt(NULL) {}

    ifStatement::ifStatement(const ifStatement &other) :
      blockStatement(other),
      condition(&(other.condition->clone())) {

      const int elifCount = (int) other.elifSmnts.size();
      for (int i = 0; i < elifCount; ++i) {
        elifStatement &elifSmnt = (elifSmnts[i]->clone()
                                   .to<elifStatement>());
        elifSmnts.push_back(&elifSmnt);
      }

      elseSmnt = (other.elseSmnt
                  ? &(other.elseSmnt->clone().to<elseStatement>())
                  : NULL);
    }

    ifStatement::~ifStatement() {
      delete condition;
    }

    void ifStatement::addElif(elifStatement &elifSmnt) {
      elifSmnts.push_back(&elifSmnt);
    }

    void ifStatement::addElse(elseStatement &elseSmnt_) {
      elseSmnt = &elseSmnt_;
    }

    statement_t& ifStatement::clone_() const {
      return *(new ifStatement(*this));
    }

    int ifStatement::type() const {
      return statementType::if_;
    }

    void ifStatement::print(printer &pout) const {
      pout.printStartIndentation();
      pout << "if (";
      pout.pushInlined(true);
      condition->print(pout);
      pout.popInlined();
      pout << ')';

      blockStatement::print(pout);

      const int elifCount = (int) elifSmnts.size();
      for (int i = 0; i < elifCount; ++i) {
        pout << *(elifSmnts[i]);
      }

      if (elseSmnt) {
        pout << (*elseSmnt);
      }
    }

    elifStatement::elifStatement(statement_t *condition_) :
      condition(condition_) {}

    elifStatement::elifStatement(const elifStatement &other) :
      blockStatement(other),
      condition(&(other.condition->clone())) {}

    elifStatement::~elifStatement() {
      delete condition;
    }

    statement_t& elifStatement::clone_() const {
      return *(new elifStatement(*this));
    }

    int elifStatement::type() const {
      return statementType::elif_;
    }

    void elifStatement::print(printer &pout) const {
      pout.printStartIndentation();
      pout << "else if (";
      pout.pushInlined(true);
      condition->print(pout);
      pout.popInlined();
      pout << ')';

      blockStatement::print(pout);
    }

    elseStatement::elseStatement() {}

    elseStatement::elseStatement(const elseStatement &other) :
      blockStatement(other) {}

    statement_t& elseStatement::clone_() const {
      return *(new elseStatement(*this));
    }

    int elseStatement::type() const {
      return statementType::else_;
    }

    void elseStatement::print(printer &pout) const {
      pout.printStartIndentation();
      pout << "else";

      blockStatement::print(pout);
    }
    //================================

    //---[ While ]----------------------
    whileStatement::whileStatement(statement_t *condition_,
                                   const bool isDoWhile_) :
      blockStatement(),
      condition(condition_),
      isDoWhile(isDoWhile_) {}

    whileStatement::whileStatement(const whileStatement &other) :
      blockStatement(other),
      condition(other.condition),
      isDoWhile(other.isDoWhile) {}

    whileStatement::~whileStatement() {
      delete condition;
    }

    statement_t& whileStatement::clone_() const {
      return *(new whileStatement(*this));
    }

    int whileStatement::type() const {
      return statementType::while_;
    }

    void whileStatement::print(printer &pout) const {
      pout.printStartIndentation();
      if (!isDoWhile) {
        pout << "while (";
        pout.pushInlined(true);
        condition->print(pout);
        pout.popInlined();
        pout << ')';
      } else {
        pout << "do";
      }

      blockStatement::print(pout);

      if (isDoWhile) {
        pout << " while (";
        pout.pushInlined(true);
        condition->print(pout);
        pout.popInlined();
        pout << ");";
      }
      pout.printEndNewline();
    }
    //==================================

    //---[ For ]------------------------
    forStatement::forStatement(statement_t &init_,
                               statement_t &check_,
                               statement_t &update_) :
      init(init_),
      check(check_),
      update(update_) {}

    forStatement::forStatement(const forStatement &other) :
      blockStatement(other),
      init(other.init.clone()),
      check(other.check.clone()),
      update(other.update.clone()) {}

    statement_t& forStatement::clone_() const {
      return *(new forStatement(*this));
    }

    int forStatement::type() const {
      return statementType::for_;
    }

    void forStatement::print(printer &pout) const {
      pout.printStartIndentation();

      pout << "for (";

      pout.pushInlined(true);
      pout << init << check << update;
      pout.popInlined();

      pout << ')';

      blockStatement::print(pout);
    }
    //==================================

    //---[ Switch ]---------------------
    switchStatement::switchStatement(statement_t *condition_) :
      blockStatement(),
      condition(condition_) {}

    switchStatement::switchStatement(const switchStatement& other) :
      blockStatement(other),
      condition(other.condition) {}

    switchStatement::~switchStatement() {
      delete condition;
    }

    statement_t& switchStatement::clone_() const {
      return *(new switchStatement(*this));
    }

    int switchStatement::type() const {
      return statementType::switch_;
    }

    void switchStatement::print(printer &pout) const {
      pout.printStartIndentation();
      pout << "switch (";
      pout.pushInlined(true);
      condition->print(pout);
      pout.popInlined();
      pout << ") {\n";

      blockStatement::print(pout);
    }
    //==================================

    //---[ Case ]-----------------------
    caseStatement::caseStatement(exprNode &value_) :
      value(&value_) {}

    caseStatement::~caseStatement() {
      delete value;
    }

    statement_t& caseStatement::clone_() const {
      return *(new caseStatement(value->clone()));
    }

    int caseStatement::type() const {
      return statementType::case_;
    }

    void caseStatement::print(printer &pout) const {
      pout.removeIndentation();

      pout.printIndentation();
      pout << "case ";
      pout.pushInlined(true);
      pout << *value;
      pout.popInlined();
      pout << ":\n";

      pout.addIndentation();
    }

    defaultStatement::defaultStatement() {}

    statement_t& defaultStatement::clone_() const {
      return *(new defaultStatement());
    }

    int defaultStatement::type() const {
      return statementType::default_;
    }

    void defaultStatement::print(printer &pout) const {
      pout.removeIndentation();

      pout.printIndentation();
      pout << "default:\n";

      pout.addIndentation();
    }
    //==================================

    //---[ Exit ]-----------------------
    continueStatement::continueStatement() :
      statement_t() {}

    statement_t& continueStatement::clone_() const {
      return *(new continueStatement());
    }

    int continueStatement::type() const {
      return statementType::continue_;
    }

    void continueStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "continue;\n";
    }

    breakStatement::breakStatement() :
      statement_t() {}

    statement_t& breakStatement::clone_() const {
      return *(new breakStatement());
    }

    int breakStatement::type() const {
      return statementType::break_;
    }

    void breakStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "break;\n";
    }

    returnStatement::returnStatement(exprNode *value_) :
      value(value_) {}

    returnStatement::returnStatement(const returnStatement &other) :
      value(NULL) {
      if (other.value) {
        value = &(other.value->clone());
      }
    }

    returnStatement::~returnStatement() {
      delete value;
    }

    statement_t& returnStatement::clone_() const {
      return *(new returnStatement(*this));
    }

    int returnStatement::type() const {
      return statementType::return_;
    }

    void returnStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "return";
      if (value) {
        pout << ' ';
        pout.pushInlined(true);
        pout << *value;
        pout.popInlined();
      }
      pout << ";\n";
    }
    //==================================
  }
}
