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
#include "builtins/types.hpp"
#include "variable.hpp"

namespace occa {
  namespace lang {
    //---[ Pragma ]--------------------
    pragmaStatement::pragmaStatement(blockStatement *up_,
                                     const pragmaToken &token_) :
      statement_t(up_),
      token(*((pragmaToken*) token_.clone())) {}

    pragmaStatement::~pragmaStatement() {
      delete &token;
    }

    statement_t& pragmaStatement::clone_() const {
      return *(new pragmaStatement(NULL, token));
    }

    int pragmaStatement::type() const {
      return statementType::pragma;
    }

    std::string& pragmaStatement::value() {
      return token.value;
    }

    const std::string& pragmaStatement::value() const {
      return token.value;
    }

    void pragmaStatement::print(printer &pout) const {
      pout << "#pragma " << token.value << '\n';
    }

    void pragmaStatement::printWarning(const std::string &message) const {
      token.printWarning(message);
    }

    void pragmaStatement::printError(const std::string &message) const {
      token.printError(message);
    }
    //==================================

    //---[ Type ]-----------------------
    functionStatement::functionStatement(blockStatement *up_,
                                         function_t &function_) :
      statement_t(up_),
      function(function_) {}

    functionStatement::~functionStatement() {
      // TODO: Add to scope with uniqueName() as the key
      function.free();
      delete &function;
    }

    statement_t& functionStatement::clone_() const {
      return *(new functionStatement(NULL, (function_t&) function.clone()));
    }

    int functionStatement::type() const {
      return statementType::function;
    }

    void functionStatement::print(printer &pout) const {
      pout.printStartIndentation();
      function.printDeclaration(pout);
      pout << ";\n";
    }

    void functionStatement::printWarning(const std::string &message) const {
      function.printWarning(message);
    }

    void functionStatement::printError(const std::string &message) const {
      function.printError(message);
    }

    functionDeclStatement::functionDeclStatement(blockStatement *up_,
                                                 function_t &function_) :
      blockStatement(up_, function_.source),
      function(function_) {}

    functionDeclStatement::functionDeclStatement(const functionDeclStatement &other) :
      blockStatement(other),
      function((function_t&) other.function.clone()) {
      updateScope(true);
    }

    statement_t& functionDeclStatement::clone_() const {
      return *(new functionDeclStatement(*this));
    }

    int functionDeclStatement::type() const {
      return statementType::functionDecl;
    }

    bool functionDeclStatement::updateScope(const bool force) {
      if (up &&
          !up->scope.add(function, force)) {
        return false;
      }
      addArgumentsToScope(force);
      return true;
    }

    void functionDeclStatement::addArgumentsToScope(const bool force) {
      const int count = (int) function.args.size();
      for (int i = 0; i < count; ++i) {
        scope.add(*(function.args[i]),
                  force);
      }
    }

    void functionDeclStatement::print(printer &pout) const {
      // Double newlines to make it look cleaner
      pout << '\n';
      pout.printStartIndentation();
      function.printDeclaration(pout);
      pout << ' ';
      blockStatement::print(pout);
    }

    classAccessStatement::classAccessStatement(blockStatement *up_,
                                               token_t *source_,
                                               const int access_) :
      statement_t(up_),
      source(token_t::clone(source_)),
      access(access_) {}

    classAccessStatement::~classAccessStatement() {
      delete source;
    }

    statement_t& classAccessStatement::clone_() const {
      return *(new classAccessStatement(NULL, source, access));
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

    void classAccessStatement::printWarning(const std::string &message) const {
      source->printWarning(message);
    }

    void classAccessStatement::printError(const std::string &message) const {
      source->printError(message);
    }
    //==================================

    //---[ Expression ]-----------------
    expressionStatement::expressionStatement(blockStatement *up_,
                                             exprNode &expr_,
                                             const bool hasSemicolon_) :
      statement_t(up_),
      expr(&expr_),
      hasSemicolon(hasSemicolon_) {}

    expressionStatement::expressionStatement(const expressionStatement &other) :
      statement_t(NULL),
      expr(other.expr->clone()),
      hasSemicolon(other.hasSemicolon) {}

    expressionStatement::~expressionStatement() {
      delete expr;
    }

    statement_t& expressionStatement::clone_() const {
      return *(new expressionStatement(*this));
    }

    int expressionStatement::type() const {
      return statementType::expression;
    }

    void expressionStatement::print(printer &pout) const {
      pout.printStartIndentation();
      pout << (*expr);
      if (hasSemicolon) {
        pout << ';';
        pout.printEndNewline();
      }
    }

    void expressionStatement::printWarning(const std::string &message) const {
      expr->startNode()->printWarning(message);
    }

    void expressionStatement::printError(const std::string &message) const {
      expr->startNode()->printError(message);
    }

    declarationStatement::declarationStatement(blockStatement *up_) :
      statement_t(up_) {}

    declarationStatement::declarationStatement(const declarationStatement &other) :
      statement_t(NULL) {
      const int count = (int) other.declarations.size();
      if (!count) {
        return;
      }
      declarations.reserve(count);
      for (int i = 0; i < count; ++i) {
        declarations.push_back(
          other.declarations[i].clone()
        );
      }
    }

    declarationStatement::~declarationStatement() {
      if (!up) {
        freeDeclarations();
        return;
      }
      const int count = (int) declarations.size();
      for (int i = 0; i < count; ++i) {
        variableDeclaration &decl = declarations[i];
        variable_t &var = *(decl.variable);
        // The scope has its own typedef copy
        // We have to delete the variable-typedef
        if (var.vartype.has(typedef_)) {
          delete &var;
        }
        declarations[i].clear();
      }
      declarations.clear();
    }

    void declarationStatement::freeDeclarations() {
      const int count = (int) declarations.size();
      for (int i = 0; i < count; ++i) {
        variable_t *var = declarations[i].variable;
        // The scope has its own typedef copy
        // We have to delete the variable-typedef
        if (up
            && up->scope.has(var->name())) {
          up->scope.remove(var->name());
          var = NULL;
        }
        delete var;
        declarations[i].clear();
      }
      declarations.clear();
    }

    statement_t& declarationStatement::clone_() const {
      return *(new declarationStatement(*this));
    }

    int declarationStatement::type() const {
      return statementType::declaration;
    }

    bool declarationStatement::addDeclaration(const variableDeclaration &decl,
                                              const bool force) {
      variable_t &var = *(decl.variable);
      bool success = true;
      if (!up) {
        delete &var;
        return false;
      }
      // Variable
      if (!var.vartype.has(typedef_)) {
        success = up->scope.add(var, force);
      } else {
        // Typedef
        typedef_t &type = *(new typedef_t(var.vartype));
        if (var.source) {
          type.setSource(*var.source);
        }

        if (var.vartype.type) {
          type.attributes = var.vartype.type->attributes;
        }
        type.attributes.insert(var.attributes.begin(),
                               var.attributes.end());

        success = up->scope.add(type, force);
        if (!success) {
          delete &type;
        }
      }
      if (success) {
        declarations.push_back(decl);
      } else {
        delete &var;
      }
      return success;
    }

    void declarationStatement::print(printer &pout) const {
      const int count = (int) declarations.size();
      if (!count) {
        return;
      }
      pout.printStartIndentation();
      declarations[0].print(pout);
      for (int i = 1; i < count; ++i) {
        pout << ", ";
        declarations[i].printAsExtra(pout);
      }
      pout << ';';
      pout.printEndNewline();
    }

    void declarationStatement::printWarning(const std::string &message) const {
      declarations[0].printWarning(message);
    }

    void declarationStatement::printError(const std::string &message) const {
      declarations[0].printError(message);
    }
    //==================================

    //---[ Goto ]-----------------------
    gotoStatement::gotoStatement(blockStatement *up_,
                                 identifierToken &labelToken_) :
      statement_t(up_),
      labelToken(labelToken_) {}

    gotoStatement::gotoStatement(const gotoStatement &other) :
      statement_t(NULL),
      labelToken(other.labelToken
                 .clone()
                 ->to<identifierToken>()) {}

    gotoStatement::~gotoStatement() {
      delete &labelToken;
    }

    statement_t& gotoStatement::clone_() const {
      return *(new gotoStatement(*this));
    }

    std::string& gotoStatement::label() {
      return labelToken.value;
    }

    const std::string& gotoStatement::label() const {
      return labelToken.value;
    }

    int gotoStatement::type() const {
      return statementType::goto_;
    }

    void gotoStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "goto " << label() << ';';
    }

    void gotoStatement::printWarning(const std::string &message) const {
      labelToken.printWarning(message);
    }

    void gotoStatement::printError(const std::string &message) const {
      labelToken.printError(message);
    }

    gotoLabelStatement::gotoLabelStatement(blockStatement *up_,
                                           identifierToken &labelToken_) :
      statement_t(up_),
      labelToken(labelToken_) {}

    gotoLabelStatement::gotoLabelStatement(const gotoLabelStatement &other) :
      statement_t(NULL),
      labelToken(other.labelToken
                 .clone()
                 ->to<identifierToken>()) {}

    gotoLabelStatement::~gotoLabelStatement() {
      delete &labelToken;
    }

    statement_t& gotoLabelStatement::clone_() const {
      return *(new gotoLabelStatement(*this));
    }

    std::string& gotoLabelStatement::label() {
      return labelToken.value;
    }

    const std::string& gotoLabelStatement::label() const {
      return labelToken.value;
    }

    int gotoLabelStatement::type() const {
      return statementType::gotoLabel;
    }

    void gotoLabelStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << label() << ":\n";
    }

    void gotoLabelStatement::printWarning(const std::string &message) const {
      labelToken.printWarning(message);
    }

    void gotoLabelStatement::printError(const std::string &message) const {
      labelToken.printError(message);
    }
    //==================================

    //---[ Namespace ]------------------
    namespaceStatement::namespaceStatement(blockStatement *up_,
                                           identifierToken &nameToken_) :
      blockStatement(up_, &nameToken_),
      nameToken(nameToken_) {}

    namespaceStatement::namespaceStatement(const namespaceStatement &other) :
      blockStatement(other),
      nameToken(other.nameToken
                .clone()
                ->to<identifierToken>()) {}

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
    ifStatement::ifStatement(blockStatement *up_,
                             token_t *source_) :
      blockStatement(up_, source_),
      condition(NULL),
      elseSmnt(NULL) {}

    ifStatement::ifStatement(const ifStatement &other) :
      blockStatement(other),
      condition(&(other.condition->clone())) {

      const int elifCount = (int) other.elifSmnts.size();
      for (int i = 0; i < elifCount; ++i) {
        elifStatement &elifSmnt = (elifSmnts[i]
                                   ->clone()
                                   .to<elifStatement>());
        elifSmnts.push_back(&elifSmnt);
      }

      elseSmnt = (other.elseSmnt
                  ? &(other.elseSmnt
                      ->clone()
                      .to<elseStatement>())
                  : NULL);
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

    elifStatement::elifStatement(blockStatement *up_,
                                 token_t *source_) :
      blockStatement(up_, source_),
      condition(NULL) {}

    elifStatement::elifStatement(const elifStatement &other) :
      blockStatement(other),
      condition(&(other.condition->clone())) {}

    elifStatement::~elifStatement() {
      delete condition;
    }

    void elifStatement::setCondition(statement_t *condition_) {
      condition = condition_;
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
      pout << ')';

      blockStatement::print(pout);
      pout.popInlined();
    }

    elseStatement::elseStatement(blockStatement *up_,
                                 token_t *source_) :
      blockStatement(up_, source_) {}

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
    whileStatement::whileStatement(blockStatement *up_,
                                   token_t *source_,
                                   const bool isDoWhile_) :
      blockStatement(up_, source_),
      condition(NULL),
      isDoWhile(isDoWhile_) {}

    whileStatement::whileStatement(const whileStatement &other) :
      blockStatement(other),
      condition(other.condition),
      isDoWhile(other.isDoWhile) {}

    whileStatement::~whileStatement() {
      delete condition;
    }

    void whileStatement::setCondition(statement_t *condition_) {
      condition = condition_;
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
        pout << ')';
      } else {
        pout << "do";
      }

      blockStatement::print(pout);

      if (isDoWhile) {
        pout.popInlined();
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
    forStatement::forStatement(blockStatement *up_,
                               token_t *source_) :
      blockStatement(up_, source_),
      init(NULL),
      check(NULL),
      update(NULL) {}

    forStatement::forStatement(const forStatement &other) :
      blockStatement(other),
      init(statement_t::clone(other.init)),
      check(statement_t::clone(other.check)),
      update(statement_t::clone(other.update)) {}

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
      pout << *init << *check << *update;
      pout << ')';

      blockStatement::print(pout);
      pout.popInlined();
    }
    //==================================

    //---[ Switch ]---------------------
    switchStatement::switchStatement(blockStatement *up_,
                                     token_t *source_) :
      blockStatement(up_, source_),
      condition(NULL) {}

    switchStatement::switchStatement(const switchStatement& other) :
      blockStatement(other),
      condition(other.condition) {}

    switchStatement::~switchStatement() {
      delete condition;
    }

    void switchStatement::setCondition(statement_t *condition_) {
      condition = condition_;
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
      pout << ") {\n";

      blockStatement::print(pout);
      pout.popInlined();
    }
    //==================================

    //---[ Case ]-----------------------
    caseStatement::caseStatement(blockStatement *up_,
                                 token_t *source_,
                                 exprNode &value_) :
      statement_t(up_),
      source(token_t::clone(source_)),
      value(&value_) {}

    caseStatement::~caseStatement() {
      delete source;
      delete value;
    }

    statement_t& caseStatement::clone_() const {
      return *(new caseStatement(NULL, source, *(value->clone())));
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

    void caseStatement::printWarning(const std::string &message) const {
      source->printWarning(message);
    }

    void caseStatement::printError(const std::string &message) const {
      source->printError(message);
    }

    defaultStatement::defaultStatement(blockStatement *up_,
                                       token_t *source_) :
      statement_t(up_),
      source(token_t::clone(source_)) {}

    defaultStatement::~defaultStatement() {
      delete source;
    }

    statement_t& defaultStatement::clone_() const {
      return *(new defaultStatement(NULL, source));
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

    void defaultStatement::printWarning(const std::string &message) const {
      source->printWarning(message);
    }

    void defaultStatement::printError(const std::string &message) const {
      source->printError(message);
    }
    //==================================

    //---[ Exit ]-----------------------
    continueStatement::continueStatement(blockStatement *up_,
                                         token_t *source_) :
      statement_t(up_),
      source(token_t::clone(source_)) {}

    continueStatement::~continueStatement() {
      delete source;
    }

    statement_t& continueStatement::clone_() const {
      return *(new continueStatement(NULL, source));
    }

    int continueStatement::type() const {
      return statementType::continue_;
    }

    void continueStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "continue;\n";
    }

    void continueStatement::printWarning(const std::string &message) const {
      source->printWarning(message);
    }

    void continueStatement::printError(const std::string &message) const {
      source->printError(message);
    }

    breakStatement::breakStatement(blockStatement *up_,
                                   token_t *source_) :
      statement_t(up_),
      source(token_t::clone(source_)) {}

    breakStatement::~breakStatement() {
      delete source;
    }

    statement_t& breakStatement::clone_() const {
      return *(new breakStatement(NULL, source));
    }

    int breakStatement::type() const {
      return statementType::break_;
    }

    void breakStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "break;\n";
    }

    void breakStatement::printWarning(const std::string &message) const {
      source->printWarning(message);
    }

    void breakStatement::printError(const std::string &message) const {
      source->printError(message);
    }

    returnStatement::returnStatement(blockStatement *up_,
                                     token_t *source_,
                                     exprNode *value_) :
      statement_t(up_),
      source(token_t::clone(source_)),
      value(value_) {}

    returnStatement::returnStatement(const returnStatement &other) :
      statement_t(NULL),
      source(token_t::clone(other.source)),
      value(NULL) {
      if (other.value) {
        value = other.value->clone();
      }
    }

    returnStatement::~returnStatement() {
      delete source;
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

    void returnStatement::printWarning(const std::string &message) const {
      source->printWarning(message);
    }

    void returnStatement::printError(const std::string &message) const {
      source->printError(message);
    }
    //==================================
  }
}
