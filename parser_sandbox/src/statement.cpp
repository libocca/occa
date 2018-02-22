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
    //---[ Pragma ]--------------------
    pragmaStatement::pragmaStatement(const std::string &line_) :
      statement_t(),
      line(line_) {}

    statement_t& pragmaStatement::clone() const {
      pragmaStatement &s = *(new pragmaStatement(line));
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

    //---[ Type ]-------------------------
    typeDeclStatement::typeDeclStatement(declarationType &declType_) :
      statement_t(),
      declType(declType_) {}


    statement_t& typeDeclStatement::clone() const {
      typeDeclStatement &s = *(new typeDeclStatement(declType));
      s.attributes = attributes;
      return s;
    }

    int typeDeclStatement::type() const {
      return statementType::typeDecl;
    }

    scope_t* typeDeclStatement::getScope() {
      if (declType.is<structureType>()) {
        return &(declType.to<structureType>().body.scope);
      }
      if (declType.is<functionType>()) {
        return &(declType.to<functionType>().body.scope);
     }
      return NULL;
    }

    void typeDeclStatement::print(printer &pout) const {
      declType.printDeclaration(pout);
    }

    classAccessStatement::classAccessStatement(const int access_) :
      statement_t(),
      access(access_) {}

    statement_t& classAccessStatement::clone() const {
      classAccessStatement &s = *(new classAccessStatement(access));
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
    expressionStatement::expressionStatement(exprNode &expression_) :
      statement_t(),
      expression(expression_) {}

    statement_t& expressionStatement::clone() const {
      expressionStatement &s = *(new expressionStatement(expression));
      s.attributes = attributes;
      return s;
    }

    int expressionStatement::type() const {
      return statementType::expression;
    }

    void expressionStatement::print(printer &pout) const {
    }

    declarationStatement::declarationStatement() :
      statement_t() {}

    statement_t& declarationStatement::clone() const {
      declarationStatement &s = *(new declarationStatement());
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
    gotoStatement::gotoStatement(const std::string &name_) :
      statement_t(),
      name(name_) {}

    statement_t& gotoStatement::clone() const {
      gotoStatement &s = *(new gotoStatement(name));
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

    gotoLabelStatement::gotoLabelStatement(const std::string &name_) :
      statement_t(),
      name(name_) {}

    statement_t& gotoLabelStatement::clone() const {
      gotoLabelStatement &s = *(new gotoLabelStatement(name));
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
    namespaceStatement::namespaceStatement(const std::string &name_) :
      blockStatement(),
      name(name_) {}

    statement_t& namespaceStatement::clone() const {
      namespaceStatement &s = *(new namespaceStatement(name));
      s.attributes = attributes;
      return s;
    }

    int namespaceStatement::type() const {
      return statementType::namespace_;
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
    whileStatement::whileStatement(statement_t &check_,
                                   const bool isDoWhile_) :
      blockStatement(),
      check(check_),
      isDoWhile(isDoWhile_) {}

    statement_t& whileStatement::clone() const {
      whileStatement &s = *(new whileStatement(check.clone(), isDoWhile));
      s.attributes = attributes;
      return s;
    }

    int whileStatement::type() const {
      return statementType::while_;
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
    forStatement::forStatement(statement_t &init_,
                               statement_t &check_,
                               statement_t &update_) :
      blockStatement(),
      init(init_),
      check(check_),
      update(update_) {}

    statement_t& forStatement::clone() const {
      forStatement &s = *(new forStatement(init.clone(),
                                           check.clone(),
                                           update.clone()));
      s.attributes = attributes;
      return s;
    }

    int forStatement::type() const {
      return statementType::for_;
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
    switchStatement::switchStatement(statement_t &value_) :
      blockStatement(),
      value(value_) {}

    statement_t& switchStatement::clone() const {
      switchStatement &s = *(new switchStatement(value.clone()));
      s.attributes = attributes;
      return s;
    }

    int switchStatement::type() const {
      return statementType::switch_;
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
    caseStatement::caseStatement(statement_t &value_) :
      statement_t(),
      value(value_) {}

    statement_t& caseStatement::clone() const {
      caseStatement &s = *(new caseStatement(value.clone()));
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
    continueStatement::continueStatement() :
      statement_t() {}

    statement_t& continueStatement::clone() const {
      continueStatement &s = *(new continueStatement());
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

    breakStatement::breakStatement() :
      statement_t() {}

    statement_t& breakStatement::clone() const {
      breakStatement &s = *(new breakStatement());
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

    returnStatement::returnStatement(statement_t &value_) :
      statement_t(),
      value(value_) {}

    statement_t& returnStatement::clone() const {
      returnStatement &s = *(new returnStatement(value.clone()));
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
