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
#include <occa/lang/expression.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/type.hpp>
#include <occa/lang/builtins/types.hpp>
#include <occa/lang/variable.hpp>

namespace occa {
  namespace lang {
    //---[ Preprocessor ]---------------
    directiveStatement::directiveStatement(blockStatement *up_,
                                           const directiveToken &token_) :
      statement_t(up_, &token_),
      token((directiveToken&) *source) {}

    directiveStatement::directiveStatement(blockStatement *up_,
                                           const directiveStatement &other) :
      statement_t(up_, other),
      token((directiveToken&) *source) {}

    directiveStatement::~directiveStatement() {}

    statement_t& directiveStatement::clone_(blockStatement *up_) const {
      return *(new directiveStatement(up_, *this));
    }

    int directiveStatement::type() const {
      return statementType::directive;
    }

    std::string directiveStatement::statementName() const {
      return "directive";
    }

    std::string& directiveStatement::value() {
      return token.value;
    }

    const std::string& directiveStatement::value() const {
      return token.value;
    }

    void directiveStatement::print(printer &pout) const {
      pout << '#' << token.value << '\n';
    }

    pragmaStatement::pragmaStatement(blockStatement *up_,
                                     const pragmaToken &token_) :
      statement_t(up_, &token_),
      token((pragmaToken&) *source) {}

    pragmaStatement::pragmaStatement(blockStatement *up_,
                                     const pragmaStatement &other) :
      statement_t(up_, other),
      token((pragmaToken&) *source) {}

    pragmaStatement::~pragmaStatement() {}

    statement_t& pragmaStatement::clone_(blockStatement *up_) const {
      return *(new pragmaStatement(up_, *this));
    }

    int pragmaStatement::type() const {
      return statementType::pragma;
    }

    std::string pragmaStatement::statementName() const {
      return "pragma";
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
    //==================================

    //---[ Type ]-----------------------
    functionStatement::functionStatement(blockStatement *up_,
                                         function_t &function_) :
      statement_t(up_, function_.source),
      function(function_) {}

    functionStatement::functionStatement(blockStatement *up_,
                                         const functionStatement &other) :
      statement_t(up_, other),
      function((function_t&) other.function.clone()) {}

    functionStatement::~functionStatement() {
      // TODO: Add to scope with uniqueName() as the key
      function.free();
      delete &function;
    }

    statement_t& functionStatement::clone_(blockStatement *up_) const {
      return *(new functionStatement(up_, *this));
    }

    int functionStatement::type() const {
      return statementType::function;
    }

    std::string functionStatement::statementName() const {
      return "function";
    }

    void functionStatement::print(printer &pout) const {
      // Double newlines to make it look cleaner
      pout << '\n';
      pout.printStartIndentation();
      function.printDeclaration(pout);
      pout << ";\n";
    }

    functionDeclStatement::functionDeclStatement(blockStatement *up_,
                                                 function_t &function_) :
      blockStatement(up_, function_.source),
      function(function_) {}

    functionDeclStatement::functionDeclStatement(blockStatement *up_,
                                                 const functionDeclStatement &other) :
      blockStatement(up_, other),
      function((function_t&) other.function.clone()) {
      updateScope(true);
    }

    statement_t& functionDeclStatement::clone_(blockStatement *up_) const {
      return *(new functionDeclStatement(up_, *this));
    }

    int functionDeclStatement::type() const {
      return statementType::functionDecl;
    }

    std::string functionDeclStatement::statementName() const {
      return "function declaration";
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
      statement_t(up_, source_),
      access(access_) {}

    classAccessStatement::classAccessStatement(blockStatement *up_,
                                               const classAccessStatement &other) :
      statement_t(up_, other),
      access(other.access) {}

    classAccessStatement::~classAccessStatement() {}

    statement_t& classAccessStatement::clone_(blockStatement *up_) const {
      return *(new classAccessStatement(up_, *this));
    }

    int classAccessStatement::type() const {
      return statementType::classAccess;
    }

    std::string classAccessStatement::statementName() const {
      return "class access";
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
    expressionStatement::expressionStatement(blockStatement *up_,
                                             exprNode &expr_,
                                             const bool hasSemicolon_) :
      statement_t(up_, expr_.startNode()->token),
      expr(&expr_),
      hasSemicolon(hasSemicolon_) {}

    expressionStatement::expressionStatement(blockStatement *up_,
                                             const expressionStatement &other) :
      statement_t(up_, other),
      expr(other.expr->clone()),
      hasSemicolon(other.hasSemicolon) {}

    expressionStatement::~expressionStatement() {
      delete expr;
    }

    statement_t& expressionStatement::clone_(blockStatement *up_) const {
      return *(new expressionStatement(up_, *this));
    }

    int expressionStatement::type() const {
      return statementType::expression;
    }

    std::string expressionStatement::statementName() const {
      return "expression";
    }

    void expressionStatement::print(printer &pout) const {
      pout.printStartIndentation();
      pout << (*expr);
      if (hasSemicolon) {
        pout << ';';
        pout.printEndNewline();
      }
    }

    declarationStatement::declarationStatement(blockStatement *up_,
                                               token_t *source_) :
      statement_t(up_, source_) {}

    declarationStatement::declarationStatement(blockStatement *up_,
                                               const declarationStatement &other) :
      statement_t(up_, other) {
      const int count = (int) other.declarations.size();
      if (!count) {
        return;
      }
      declarations.reserve(count);
      for (int i = 0; i < count; ++i) {
        addDeclaration(other.declarations[i].clone());
      }
    }

    declarationStatement::~declarationStatement() {
      if (up) {
        clearDeclarations();
      } else {
        freeDeclarations();
      }
    }

    void declarationStatement::clearDeclarations() {
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

    statement_t& declarationStatement::clone_(blockStatement *up_) const {
      return *(new declarationStatement(up_, *this));
    }

    int declarationStatement::type() const {
      return statementType::declaration;
    }

    std::string declarationStatement::statementName() const {
      return "declaration";
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
    //==================================

    //---[ Goto ]-----------------------
    gotoStatement::gotoStatement(blockStatement *up_,
                                 identifierToken &labelToken_) :
      statement_t(up_, &labelToken_),
      labelToken((identifierToken&) *source) {}

    gotoStatement::gotoStatement(blockStatement *up_,
                                 const gotoStatement &other) :
      statement_t(up_, other),
      labelToken((identifierToken&) *source) {}

    gotoStatement::~gotoStatement() {}

    statement_t& gotoStatement::clone_(blockStatement *up_) const {
      return *(new gotoStatement(up_, *this));
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

    std::string gotoStatement::statementName() const {
      return "goto";
    }

    void gotoStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "goto " << label() << ';';
    }

    gotoLabelStatement::gotoLabelStatement(blockStatement *up_,
                                           identifierToken &labelToken_) :
      statement_t(up_, &labelToken_),
      labelToken((identifierToken&) *source) {}

    gotoLabelStatement::gotoLabelStatement(blockStatement *up_,
                                           const gotoLabelStatement &other) :
      statement_t(up_, other),
      labelToken((identifierToken&) *source) {}

    gotoLabelStatement::~gotoLabelStatement() {}

    statement_t& gotoLabelStatement::clone_(blockStatement *up_) const {
      return *(new gotoLabelStatement(up_, *this));
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

    std::string gotoLabelStatement::statementName() const {
      return "goto label";
    }

    void gotoLabelStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << label() << ":\n";
    }
    //==================================

    //---[ Namespace ]------------------
    namespaceStatement::namespaceStatement(blockStatement *up_,
                                           identifierToken &nameToken_) :
      blockStatement(up_, &nameToken_),
      nameToken((identifierToken&) *source) {}

    namespaceStatement::namespaceStatement(blockStatement *up_,
                                           const namespaceStatement &other) :
      blockStatement(up_, other),
      nameToken((identifierToken&) *source) {}

    namespaceStatement::~namespaceStatement() {}

    statement_t& namespaceStatement::clone_(blockStatement *up_) const {
      return *(new namespaceStatement(up_, *this));
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

    std::string namespaceStatement::statementName() const {
      return "namespace";
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

    elifStatement::elifStatement(blockStatement *up_,
                                 const elifStatement &other) :
      blockStatement(up_, other.source),
      condition(&(other.condition->clone(this))) {
      copyFrom(other);
    }

    elifStatement::~elifStatement() {
      delete condition;
    }

    void elifStatement::setCondition(statement_t *condition_) {
      condition = condition_;
    }

    statement_t& elifStatement::clone_(blockStatement *up_) const {
      return *(new elifStatement(up_, *this));
    }

    int elifStatement::type() const {
      return statementType::elif_;
    }

    std::string elifStatement::statementName() const {
      return "else if";
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

    elseStatement::elseStatement(blockStatement *up_,
                                 const elseStatement &other) :
      blockStatement(up_, other) {}

    statement_t& elseStatement::clone_(blockStatement *up_) const {
      return *(new elseStatement(up_, *this));
    }

    int elseStatement::type() const {
      return statementType::else_;
    }

    std::string elseStatement::statementName() const {
      return "else";
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

    whileStatement::whileStatement(blockStatement *up_,
                                   const whileStatement &other) :
      blockStatement(up_, other.source),
      condition(statement_t::clone(up_, other.condition)),
      isDoWhile(other.isDoWhile) {
      copyFrom(other);
    }

    whileStatement::~whileStatement() {
      delete condition;
    }

    void whileStatement::setCondition(statement_t *condition_) {
      condition = condition_;
    }

    statement_t& whileStatement::clone_(blockStatement *up_) const {
      return *(new whileStatement(up_, *this));
    }

    int whileStatement::type() const {
      return statementType::while_;
    }

    std::string whileStatement::statementName() const {
      return isDoWhile ? "do while" : "while";
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
      } else {
        pout.popInlined();
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

    switchStatement::switchStatement(blockStatement *up_,
                                     const switchStatement& other) :
      blockStatement(up_, other.source),
      condition(&(other.condition->clone(this))) {
      copyFrom(other);
    }

    switchStatement::~switchStatement() {
      delete condition;
    }

    void switchStatement::setCondition(statement_t *condition_) {
      condition = condition_;
    }

    statement_t& switchStatement::clone_(blockStatement *up_) const {
      return *(new switchStatement(up_, *this));
    }

    int switchStatement::type() const {
      return statementType::switch_;
    }

    std::string switchStatement::statementName() const {
      return "switch";
    }

    void switchStatement::print(printer &pout) const {
      pout.printStartIndentation();
      pout << "switch (";
      pout.pushInlined(true);
      condition->print(pout);
      pout << ')';

      blockStatement::print(pout);
      pout.popInlined();
    }
    //==================================

    //---[ Case ]-----------------------
    caseStatement::caseStatement(blockStatement *up_,
                                 token_t *source_,
                                 exprNode &value_) :
      statement_t(up_, source_),
      value(&value_) {}

    caseStatement::caseStatement(blockStatement *up_,
                                 const caseStatement &other) :
      statement_t(up_, other),
      value(other.value->clone()) {}

    caseStatement::~caseStatement() {
      delete value;
    }

    statement_t& caseStatement::clone_(blockStatement *up_) const {
      return *(new caseStatement(up_, *this));
    }

    int caseStatement::type() const {
      return statementType::case_;
    }

    std::string caseStatement::statementName() const {
      return "case";
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

    defaultStatement::defaultStatement(blockStatement *up_,
                                       token_t *source_) :
      statement_t(up_, source_) {}

    defaultStatement::defaultStatement(blockStatement *up_,
                                       const defaultStatement &other) :
      statement_t(up_, other) {}

    defaultStatement::~defaultStatement() {}

    statement_t& defaultStatement::clone_(blockStatement *up_) const {
      return *(new defaultStatement(up_, *this));
    }

    int defaultStatement::type() const {
      return statementType::default_;
    }

    std::string defaultStatement::statementName() const {
      return "default";
    }

    void defaultStatement::print(printer &pout) const {
      pout.removeIndentation();

      pout.printIndentation();
      pout << "default:\n";

      pout.addIndentation();
    }
    //==================================

    //---[ Exit ]-----------------------
    continueStatement::continueStatement(blockStatement *up_,
                                         token_t *source_) :
      statement_t(up_, source_) {}

    continueStatement::continueStatement(blockStatement *up_,
                                         const continueStatement &other) :
      statement_t(up_, other) {}

    continueStatement::~continueStatement() {}

    statement_t& continueStatement::clone_(blockStatement *up_) const {
      return *(new continueStatement(up_, *this));
    }

    int continueStatement::type() const {
      return statementType::continue_;
    }

    std::string continueStatement::statementName() const {
      return "continue";
    }

    void continueStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "continue;\n";
    }

    breakStatement::breakStatement(blockStatement *up_,
                                   token_t *source_) :
      statement_t(up_, source_) {}

    breakStatement::breakStatement(blockStatement *up_,
                                   const breakStatement &other) :
      statement_t(up_, other) {}

    breakStatement::~breakStatement() {}

    statement_t& breakStatement::clone_(blockStatement *up_) const {
      return *(new breakStatement(up_, *this));
    }

    int breakStatement::type() const {
      return statementType::break_;
    }

    std::string breakStatement::statementName() const {
      return "break";
    }

    void breakStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "break;\n";
    }

    returnStatement::returnStatement(blockStatement *up_,
                                     token_t *source_,
                                     exprNode *value_) :
      statement_t(up_, source_),
      value(value_) {}

    returnStatement::returnStatement(blockStatement *up_,
                                     const returnStatement &other) :
      statement_t(up_, other),
      value(exprNode::clone(other.value)) {}

    returnStatement::~returnStatement() {
      delete value;
    }

    statement_t& returnStatement::clone_(blockStatement *up_) const {
      return *(new returnStatement(up_, *this));
    }

    int returnStatement::type() const {
      return statementType::return_;
    }

    std::string returnStatement::statementName() const {
      return "return";
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
