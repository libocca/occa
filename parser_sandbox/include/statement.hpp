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
#ifndef OCCA_PARSER_STATEMENT_HEADER2
#define OCCA_PARSER_STATEMENT_HEADER2

#include "macro.hpp"
#include "context.hpp"
#include "scope.hpp"

namespace occa {
  namespace lang {
    class emptyStatement;
    class directiveStatement;
    class blockStatement;
    class typeDeclStatement;
    class classAccessStatement;
    class expressionStatement;
    class declarationStatement;
    class gotoStatement;
    class gotoLabelStatement;
    class namespaceStatement;
    class whileStatement;
    class forStatement;
    class switchStatement;
    class caseStatement;
    class continueStatement;
    class breakStatement;
    class returnStatement;

    class statementType {
    public:
      static const int empty       = (1 << 0);
      static const int directive   = (1 << 1);
      static const int block       = (1 << 2);
      static const int typeDecl    = (1 << 3);
      static const int classAccess = (1 << 4);
      static const int expression  = (1 << 5);
      static const int declaration = (1 << 6);
      static const int goto_       = (1 << 7);
      static const int gotoLabel   = (1 << 8);
      static const int namespace_  = (1 << 9);
      static const int while_      = (1 << 10);
      static const int for_        = (1 << 11);
      static const int switch_     = (1 << 12);
      static const int case_       = (1 << 13);
      static const int continue_   = (1 << 14);
      static const int break_      = (1 << 15);
      static const int return_     = (1 << 16);
    };

    class statement {
    public:
      statement *up;
      context &ctx;
      scope_t scope;

      statement(context &ctx_);

      virtual ~statement();

      template <class TM>
      inline bool is() const {
        return (dynamic_cast<const TM*>(this) != NULL);
      }

      template <class TM>
      inline TM& to() {
        TM *ptr = dynamic_cast<TM*>(this);
        OCCA_ERROR("Unable to cast statement::to",
                   ptr != NULL);
        return *ptr;
      }

      template <class TM>
      inline const TM& to() const {
        const TM *ptr = dynamic_cast<const TM*>(this);
        OCCA_ERROR("Unable to cast statement::to",
                   ptr != NULL);
        return *ptr;
      }

      virtual statement& clone() const = 0;
      virtual int type() const = 0;

      virtual bool hasScope() const;

      // Creation methods
      emptyStatement       newEmptyStatement();
      directiveStatement   newDirectiveStatement(macro_t &macro);
      blockStatement       newBlockStatement();
      typeDeclStatement    newTypeDeclarationStatement(declarationType &declType);
      classAccessStatement newClassAccessStatement(const int access);
      expressionStatement  newExpressionStatement(exprNode &expression);
      declarationStatement newDeclarationStatement();
      gotoStatement        newGotoStatement(const std::string &name);
      gotoLabelStatement   newGotoLabelStatement(const std::string &name);
      namespaceStatement   newNamespaceStatement(const std::string &name);
      whileStatement       newWhileStatement(statement &check);
      whileStatement       newDoWhileStatement(statement &check);
      forStatement         newForStatement(statement &init,
                                           statement &check,
                                           statement &update);
      switchStatement      newSwitchStatement(statement &value);
      caseStatement        newCaseStatement(statement &value);
      continueStatement    newContinueStatement();
      breakStatement       newBreakStatement();
      returnStatement      newReturnStatement(statement &value);

      virtual void print(printer &pout) const = 0;

      std::string toString() const;
      operator std::string() const;
      void print() const;
    };

    //---[ Empty ]------------------------
    class emptyStatement : public statement {
    public:
      emptyStatement(context &ctx_);

      virtual statement& clone() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //====================================

    //---[ Directive ]--------------------
    class directiveStatement : public statement {
    public:
      macro_t &macro;

      directiveStatement(context &ctx_,
                         macro_t &macro_);

      virtual statement& clone() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //====================================

    //---[ Block ]------------------------
    class blockStatement : public statement {
    public:
      statementPtrVector children;

      blockStatement(context &ctx_);

      void addChild(statement &child);
      void clearChildren();

      virtual statement& clone() const;
      virtual int type() const;

      virtual bool hasScope() const;

      virtual void print(printer &pout) const;
      void printChildren(printer &pout) const;
    };
    //====================================

    //---[ Type ]-------------------------
    class typeDeclStatement : public statement {
    public:
      declarationType &declType;

      typeDeclStatement(context &ctx_,
                        declarationType &declType_);

      virtual statement& clone() const;
      virtual int type() const;

      virtual bool hasScope() const;

      virtual void print(printer &pout) const;
    };

    class classAccessStatement : public statement {
    public:
      int access;

      classAccessStatement(context &ctx_,
                           const int access_);

      virtual statement& clone() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //====================================

    //---[ Expression ]-------------------
    class expressionStatement : public statement {
    public:
      exprNode &expression;

      expressionStatement(context &ctx_,
                          exprNode &expression_);

      virtual statement& clone() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class declarationStatement : public statement {
    public:
      declarationStatement(context &ctx_);

      virtual statement& clone() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //====================================

    //---[ Goto ]-------------------------
    class gotoStatement : public statement {
    public:
      std::string name;

      gotoStatement(context &ctx_,
                    const std::string &name_);

      virtual statement& clone() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class gotoLabelStatement : public statement {
    public:
      std::string name;

      gotoLabelStatement(context &ctx_,
                         const std::string &name_);

      virtual statement& clone() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //====================================

    //---[ Namespace ]--------------------
    class namespaceStatement : public blockStatement {
    public:
      std::string name;

      namespaceStatement(context &ctx_,
                         const std::string &name_);

      virtual bool hasScope() const;

      virtual statement& clone() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //====================================

    //---[ While ]------------------------
    class whileStatement : public blockStatement {
    public:
      statement &check;
      bool isDoWhile;

      whileStatement(context &ctx_,
                     statement &check_,
                     const bool isDoWhile_ = false);

      virtual statement& clone() const;
      virtual int type() const;

      virtual bool hasScope() const;

      virtual void print(printer &pout) const;
    };
    //====================================

    //---[ For ]--------------------------
    class forStatement : public blockStatement {
    public:
      statement &init, &check, &update;

      forStatement(context &ctx_,
                   statement &init_,
                   statement &check_,
                   statement &update_);

      virtual statement& clone() const;
      virtual int type() const;

      virtual bool hasScope() const;

      virtual void print(printer &pout) const;
    };
    //====================================

    //---[ Switch ]-----------------------
    class switchStatement : public blockStatement {
    public:
      statement &value;

      switchStatement(context &ctx_,
                      statement &value_);

      virtual statement& clone() const;
      virtual int type() const;

      virtual bool hasScope() const;

      virtual void print(printer &pout) const;
    };
    //====================================

    //---[ Case ]-------------------------
    class caseStatement : public statement {
    public:
      statement &value;

      caseStatement(context &ctx_,
                    statement &value_);

      virtual statement& clone() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //====================================

    //---[ Exit ]-------------------------
    class continueStatement : public statement {
    public:
      continueStatement(context &ctx_);

      virtual statement& clone() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class breakStatement : public statement {
    public:
      breakStatement(context &ctx_);

      virtual statement& clone() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class returnStatement : public statement {
    public:
      statement &value;

      returnStatement(context &ctx_,
                      statement &value_);

      virtual statement& clone() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //====================================
  }
}

#endif
