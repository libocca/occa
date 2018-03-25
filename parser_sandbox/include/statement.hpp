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
#ifndef OCCA_LANG_STATEMENT_HEADER
#define OCCA_LANG_STATEMENT_HEADER

#include <vector>

#include "baseStatement.hpp"
#include "scope.hpp"
#include "token.hpp"
#include "trie.hpp"
#include "type.hpp"

namespace occa {
  namespace lang {
    class ifStatement;
    class elifStatement;
    class elseStatement;
    class variableDeclaration;

    typedef std::vector<elifStatement*>      elifStatementVector;
    typedef std::vector<variableDeclaration> variableDeclarationVector;

    //---[ Pragma ]---------------------
    class pragmaStatement : public statement_t {
    public:
      pragmaToken &token;

      pragmaStatement(pragmaToken &token_);
      ~pragmaStatement();

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ Type ]-----------------------
    // TODO: Type declaration

    class classAccessStatement : public statement_t {
    public:
      int access;

      classAccessStatement(const int access_);

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ Expression ]-----------------
    class expressionStatement : public statement_t {
    public:
      exprNode *root;

      expressionStatement(exprNode &root_);
      ~expressionStatement();

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class declarationStatement : public statement_t {
    public:
      variableDeclarationVector declarations;

      declarationStatement();
      declarationStatement(const declarationStatement &other);

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ Goto ]-----------------------
    class gotoStatement : public statement_t {
    public:
      std::string name;

      gotoStatement(const std::string &name_);

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class gotoLabelStatement : public statement_t {
    public:
      std::string name;

      gotoLabelStatement(const std::string &name_);

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ Namespace ]------------------
    class namespaceStatement : public blockStatement {
    public:
      std::string name;

      namespaceStatement(const std::string &name_);
      namespaceStatement(const namespaceStatement &other);

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ If ]-------------------------
    class ifStatement : public blockStatement {
    public:
      exprNode &condition;

      elifStatementVector elifSmnts;
      elseStatement *elseSmnt;

      ifStatement(exprNode &condition_);
      ifStatement(const ifStatement &other);

      void addElif(elifStatement &elifSmnt);

      void addElse(elseStatement &elseSmnt_);

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class elifStatement : public blockStatement {
    public:
      exprNode &condition;

      elifStatement(exprNode &condition);
      elifStatement(const elifStatement &other);

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class elseStatement : public blockStatement {
    public:
      elseStatement();
      elseStatement(const elseStatement &other);

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ For ]------------------------
    class forStatement : public blockStatement {
    public:
      statement_t &init, &check, &update;

      forStatement(statement_t &init_,
                   statement_t &check_,
                   statement_t &update_);

      forStatement(const forStatement &other);

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ While ]----------------------
    class whileStatement : public blockStatement {
    public:
      statement_t &check;
      bool isDoWhile;

      whileStatement(statement_t &check_,
                     const bool isDoWhile_ = false);

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ Switch ]---------------------
    class switchStatement : public blockStatement {
    public:
      statement_t &value;

      switchStatement(statement_t &value_);
      switchStatement(const switchStatement& other);

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ Case ]-----------------------
    class caseStatement : public statement_t {
    public:
      exprNode *value;

      caseStatement(exprNode &value_);
      ~caseStatement();

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class defaultStatement : public statement_t {
    public:
      defaultStatement();

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ Exit ]-----------------------
    class continueStatement : public statement_t {
    public:
      continueStatement();

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class breakStatement : public statement_t {
    public:
      breakStatement();

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class returnStatement : public statement_t {
    public:
      statement_t &value;

      returnStatement(statement_t &value_);

      virtual statement_t& clone_() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //==================================
  }
}

#endif
