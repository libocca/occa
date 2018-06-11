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
#ifndef OCCA_LANG_BASESTATEMENT_HEADER
#define OCCA_LANG_BASESTATEMENT_HEADER

#include <vector>

#include <occa/lang/printer.hpp>
#include <occa/lang/scope.hpp>
#include <occa/lang/attribute.hpp>

namespace occa {
  namespace lang {
    class statement_t;
    class blockStatement;

    typedef std::vector<statement_t*> statementPtrVector;

    namespace statementType {
      extern const int none;
      extern const int all;

      extern const int empty;

      extern const int directive;
      extern const int pragma;

      extern const int block;
      extern const int namespace_;

      extern const int typeDecl;
      extern const int function;
      extern const int functionDecl;
      extern const int classAccess;

      extern const int expression;
      extern const int declaration;

      extern const int goto_;
      extern const int gotoLabel;

      extern const int if_;
      extern const int elif_;
      extern const int else_;
      extern const int for_;
      extern const int while_;
      extern const int switch_;
      extern const int case_;
      extern const int default_;
      extern const int continue_;
      extern const int break_;

      extern const int return_;

      extern const int attribute;
    }

    class statement_t {
    public:
      blockStatement *up;
      attributeTokenMap attributes;

      statement_t(blockStatement *up_);

      virtual ~statement_t();

      template <class TM>
      inline bool is() const {
        return (dynamic_cast<const TM*>(this) != NULL);
      }

      template <class TM>
      inline TM& to() {
        TM *ptr = dynamic_cast<TM*>(this);
        OCCA_ERROR("Unable to cast statement_t::to",
                   ptr != NULL);
        return *ptr;
      }

      template <class TM>
      inline const TM& to() const {
        const TM *ptr = dynamic_cast<const TM*>(this);
        OCCA_ERROR("Unable to cast statement_t::to",
                   ptr != NULL);
        return *ptr;
      }

      statement_t& clone(blockStatement *up_ = NULL) const;
      virtual statement_t& clone_(blockStatement *up_) const = 0;

      static statement_t* clone(blockStatement *up_,
                                statement_t *smnt);

      virtual int type() const = 0;

      virtual bool inScope(const std::string &name);
      virtual keyword_t& getScopeKeyword(const std::string &name);

      void addAttribute(const attributeToken_t &attribute);
      bool hasAttribute(const std::string &attr) const;

      int childIndex() const;
      void removeFromParent();

      virtual void print(printer &pout) const = 0;

      std::string toString() const;
      operator std::string() const;
      void debugPrint() const;

      virtual void printWarning(const std::string &message) const = 0;
      virtual void printError(const std::string &message) const = 0;
    };

    printer& operator << (printer &pout,
                          const statement_t &smnt);

    //---[ Empty ]----------------------
    class emptyStatement : public statement_t {
    public:
      token_t *source;
      bool hasSemicolon;

      emptyStatement(blockStatement *up_,
                     token_t *source_,
                     const bool hasSemicolon_ = true);

      ~emptyStatement();

      virtual statement_t& clone_(blockStatement *up_) const;
      virtual int type() const;

      virtual void print(printer &pout) const;

      virtual void printWarning(const std::string &message) const;
      virtual void printError(const std::string &message) const;
    };
    //==================================

    //---[ Block ]------------------------
    class blockStatement : public statement_t {
    public:
      token_t *source;
      statementPtrVector children;
      scope_t scope;

      blockStatement(blockStatement *up_,
                     token_t *source_);
      blockStatement(blockStatement *up_,
                     const blockStatement &other);
      virtual ~blockStatement();

      void copyFrom(const blockStatement &other);

      virtual statement_t& clone_(blockStatement *up_) const;
      virtual int type() const;

      virtual bool inScope(const std::string &name);
      virtual keyword_t& getScopeKeyword(const std::string &name);

      statement_t* operator [] (const int index);

      int size() const;

      void add(statement_t &child);

      bool add(statement_t &child,
               const int index);

      bool addFirst(statement_t &child);

      bool addLast(statement_t &child);

      bool addBefore(statement_t &child,
                     statement_t &newChild);

      bool addAfter(statement_t &child,
                    statement_t &newChild);

      void remove(statement_t &child);

      void set(statement_t &child);

      void swap(blockStatement &other);

      void clear();

      exprNode* replaceIdentifiers(exprNode *expr);

      virtual void print(printer &pout) const;
      void printChildren(printer &pout) const;

      virtual void printWarning(const std::string &message) const;
      virtual void printError(const std::string &message) const;
    };
    //====================================
  }
}

#endif
