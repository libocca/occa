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
#include "baseStatement.hpp"
#include "type.hpp"

namespace occa {
  namespace lang {
    namespace statementType {
      const int none         = (1 << 0);
      const int empty        = (1 << 1);

      const int pragma       = (1 << 2);

      const int block        = (1 << 3);
      const int namespace_   = (1 << 4);

      const int typeDecl     = (1 << 5);
      const int function     = (1 << 6);
      const int functionDecl = (1 << 7);
      const int classAccess  = (1 << 8);

      const int expression   = (1 << 9);
      const int declaration  = (1 << 10);

      const int goto_        = (1 << 11);
      const int gotoLabel    = (1 << 12);

      const int if_          = (1 << 13);
      const int elif_        = (1 << 14);
      const int else_        = (1 << 15);
      const int for_         = (1 << 16);
      const int while_       = (1 << 17);
      const int switch_      = (1 << 18);
      const int case_        = (1 << 19);
      const int default_     = (1 << 20);
      const int continue_    = (1 << 21);
      const int break_       = (1 << 22);

      const int return_      = (1 << 23);

      const int attribute    = (1 << 24);
    }

    statement_t::statement_t(blockStatement *up_) :
      up(up_),
      attributes() {}

    statement_t::~statement_t() {}

    statement_t& statement_t::clone() const {
      statement_t &s = clone_();
      s.attributes = attributes;
      return s;
    }

    bool statement_t::inScope(const std::string &name) {
      if (up) {
        return up->inScope(name);
      }
      return false;
    }

    scopeKeyword_t statement_t::getScopeKeyword(const std::string &name) {
      if (up) {
        return up->getScopeKeyword(name);
      }
      return scopeKeyword_t();
    }

    void statement_t::addAttribute(attribute_t &attribute) {
      // TODO: Warning if attribute already exists
      // Override last attribute by default
      attributes.push_back(&attribute);
    }

    std::string statement_t::toString() const {
      std::stringstream ss;
      printer pout(ss);
      pout << (*this);
      return ss.str();
    }

    statement_t::operator std::string() const {
      return toString();
    }

    void statement_t::print() const {
      std::cout << toString();
    }

    printer& operator << (printer &pout,
                          const statement_t &smnt) {
      smnt.print(pout);
      return pout;
    }

    //---[ Empty ]------------------------
    emptyStatement::emptyStatement(blockStatement *up_) :
      statement_t(up_) {}

    statement_t& emptyStatement::clone_() const {
      return *(new emptyStatement(NULL));
    }

    int emptyStatement::type() const {
      return statementType::empty;
    }

    void emptyStatement::print(printer &pout) const {
      pout << ';';
    }
    //====================================

    //---[ Block ]------------------------
    blockStatement::blockStatement(blockStatement *up_) :
      statement_t(up_) {}

    blockStatement::blockStatement(const blockStatement &other) :
      statement_t(NULL) {
      attributes = other.attributes;
      const int childCount = (int) other.children.size();
      for (int i = 0; i < childCount; ++i) {
        add(other.children[i]->clone());
      }
    }

    blockStatement::~blockStatement() {
      clear();
    }

    statement_t& blockStatement::clone_() const {
      return *(new blockStatement(*this));
    }

    int blockStatement::type() const {
      return statementType::block;
    }

    bool blockStatement::inScope(const std::string &name) {
      if (scope.has(name)) {
        return true;
      }
      return (up
              ? up->inScope(name)
              : false);
    }

    scopeKeyword_t blockStatement::getScopeKeyword(const std::string &name) {
      scopeKeyword_t keyword = scope.get(name);
      if (keyword.exists()) {
        return keyword;
      }
      if (up) {
        return up->getScopeKeyword(name);
      }
      return scopeKeyword_t();
    }

    statement_t* blockStatement::operator [] (const int index) {
      if ((index < 0) ||
          (index >= (int) children.size())) {
        return NULL;
      }
      return children[index];
    }

    int blockStatement::size() const {
      return (int) children.size();
    }

    void blockStatement::add(statement_t &child) {
      children.push_back(&child);
      child.up = this;
    }

    void blockStatement::set(statement_t &child) {
      blockStatement *body = dynamic_cast<blockStatement*>(&child);
      if (!body) {
        add(child);
        return;
      }

      // Swap body contents
      children = body->children;
      scope    = body->scope;

      // Update children's up statement
      const int childCount = (int) children.size();
      for (int i = 0; i < childCount; ++i) {
        children[i]->up = this;
      }

      // Clear old body statement
      body->children.clear();
      body->scope.keywordMap.clear();
      delete body;
    }

    void blockStatement::clear() {
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
        pout << *(children[i]);
      }
    }
    //====================================
  }
}
