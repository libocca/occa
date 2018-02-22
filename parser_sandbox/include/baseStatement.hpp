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
#ifndef OCCA_PARSER_BASESTATEMENT_HEADER2
#define OCCA_PARSER_BASESTATEMENT_HEADER2

#include <vector>

#include "printer.hpp"
#include "scope.hpp"
#include "trie.hpp"

namespace occa {
  namespace lang {
    class statement_t;
    class attribute_t;

    typedef std::vector<statement_t*> statementPtrVector;
    typedef std::vector<attribute_t>  attributeVector_t;

    class statementType {
    public:
      static const int none        = 0;
      static const int empty       = (1 << 0);
      static const int pragma      = (1 << 1);
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
      static const int attribute   = (1 << 17);
    };

    class statement_t {
    public:
      statement_t *up;
      attributeVector_t attributes;

      statement_t();

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

      virtual statement_t& clone() const = 0;
      virtual int type() const = 0;

      virtual scope_t* getScope();

      void addAttribute(const attribute_t &attribute);

      virtual void print(printer &pout) const = 0;

      std::string toString() const;
      operator std::string() const;
      void print() const;
    };

    //---[ Empty ]----------------------
    class emptyStatement : public statement_t {
    public:
      emptyStatement();

      virtual statement_t& clone() const;
      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ Block ]------------------------
    class blockStatement : public statement_t {
    public:
      statementPtrVector children;
      scope_t scope;

      blockStatement();

      bool hasChildren() const;
      void addChild(statement_t &child);
      void clearChildren();

      virtual statement_t& clone() const;
      virtual int type() const;

      virtual scope_t* getScope();

      virtual void print(printer &pout) const;
      void printChildren(printer &pout) const;
    };
    //====================================
  }
}

#endif
