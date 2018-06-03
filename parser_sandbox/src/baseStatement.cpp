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
#include "token.hpp"
#include "type.hpp"
#include "builtins/transforms/fillExprIdentifiers.hpp"

namespace occa {
  namespace lang {
    namespace statementType {
      const int none         = (1 << 0);
      const int all          = -1;

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

    statement_t* statement_t::clone(statement_t *smnt) {
      if (smnt) {
        return &(smnt->clone());
      }
      return NULL;
    }

    bool statement_t::inScope(const std::string &name) {
      if (up) {
        return up->inScope(name);
      }
      return false;
    }

    keyword_t& statement_t::getScopeKeyword(const std::string &name) {
      return up->getScopeKeyword(name);
    }

    void statement_t::addAttribute(const attributeToken_t &attribute) {
      attributes[attribute.name()] = attribute;
    }

    bool statement_t::hasAttribute(const std::string &attr) const {
      return (attributes.find(attr) != attributes.end());
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

    int statement_t::childIndex() const {
      if (!up ||
          !up->is<blockStatement>()) {
        return -1;
      }
      blockStatement &upBlock = *((blockStatement*) up);
      const int childrenCount = (int) upBlock.children.size();
      for (int i = 0; i < childrenCount; ++i) {
        if (upBlock.children[i] == this) {
          return i;
        }
      }
      return -1;
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
    emptyStatement::emptyStatement(blockStatement *up_,
                                   token_t *source_,
                                   const bool hasSemicolon_) :
      statement_t(up_),
      source(token_t::clone(source_)),
      hasSemicolon(hasSemicolon_) {}

    emptyStatement::~emptyStatement() {
      delete source;
    }

    statement_t& emptyStatement::clone_() const {
      return *(new emptyStatement(NULL,
                                  source));
    }

    int emptyStatement::type() const {
      return statementType::empty;
    }

    void emptyStatement::print(printer &pout) const {
      if (hasSemicolon) {
        pout << ';';
      }
    }

    void emptyStatement::printWarning(const std::string &message) const {
      source->printWarning(message);
    }

    void emptyStatement::printError(const std::string &message) const {
      source->printError(message);
    }
    //====================================

    //---[ Block ]------------------------
    blockStatement::blockStatement(blockStatement *up_,
                                   token_t *source_) :
      statement_t(up_),
      source(token_t::clone(source_)) {}

    blockStatement::blockStatement(const blockStatement &other) :
      statement_t(NULL),
      source(token_t::clone(other.source)) {
      attributes = other.attributes;
      const int childCount = (int) other.children.size();
      for (int i = 0; i < childCount; ++i) {
        add(other.children[i]->clone());
      }
    }

    blockStatement::~blockStatement() {
      clear();
      delete source;
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

    keyword_t& blockStatement::getScopeKeyword(const std::string &name) {
      keyword_t &keyword = scope.get(name);
      if ((keyword.type() == keywordType::none)
          && up) {
        return up->getScopeKeyword(name);
      }
      return keyword;
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

    bool blockStatement::add(statement_t &child,
                             const int index) {
      const int count = (int) children.size();
      if ((index < 0) || (count < index)) {
        child.printError("Unable to add to parent with given index ["
                         + occa::toString(index) + "]");
        return false;
      }
      children.insert(children.begin() + index,
                      &child);
      child.up = this;
      return true;
    }

    bool blockStatement::addFirst(statement_t &child) {
      return add(child, 0);
    }

    bool blockStatement::addLast(statement_t &child) {
      return add(child, (int) children.size());
    }

    bool blockStatement::addBefore(statement_t &child,
                                   statement_t &newChild) {
      const int index = child.childIndex();
      if (index < 0) {
        child.printError("Not a child statement");
        printError("Expected parent of child statement");
        return false;
      }
      children.insert(children.begin() + index,
                      &newChild);
      newChild.up = this;
      return true;
    }

    bool blockStatement::addAfter(statement_t &child,
                                  statement_t &newChild) {
      const int index = child.childIndex();
      if (index < 0) {
        child.printError("Not a child statement");
        printError("Expected parent of child statement");
        return false;
      }
      children.insert(children.begin() + index + 1,
                      &newChild);
      newChild.up = this;
      return true;
    }

    void blockStatement::set(statement_t &child) {
      if (child.type() != statementType::block) {
        add(child);
        return;
      }

      blockStatement &body = (blockStatement&) child;
      swap(body);
      body.scope.moveTo(scope);
      delete &body;
    }

    void blockStatement::swap(blockStatement &other) {
      scope.swap(other.scope);
      children.swap(other.children);

      const int childCount = (int) children.size();
      for (int i = 0; i < childCount; ++i) {
        children[i]->up = this;
      }
      const int otherChildCount = (int) other.children.size();
      for (int i = 0; i < otherChildCount; ++i) {
        other.children[i]->up = &other;
      }
    }

    void blockStatement::clear() {
      const int count = (int) children.size();
      for (int i = 0; i < count; ++i) {
        delete children[i];
      }
      children.clear();
      scope.clear();
    }

    exprNode* blockStatement::replaceIdentifiers(exprNode *expr) {
      if (!expr) {
        return NULL;
      }
      transforms::fillExprIdentifiers_t replacer(this);
      return replacer.apply(*expr);
    }

    void blockStatement::print(printer &pout) const {
      bool hasChildren = children.size();
      if (!hasChildren) {
        if (up) {
          pout.printStartIndentation();
          pout << "{}\n";
        }
        return;
      }

      // Don't print { } for root statement
      if (up) {
        pout.printStartIndentation();
        pout.pushInlined(false);
        pout << "{\n";
        pout.addIndentation();
      }

      printChildren(pout);

      if (up) {
        pout.removeIndentation();
        pout.popInlined();
        pout.printNewline();
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

    void blockStatement::printWarning(const std::string &message) const {
      source->printWarning(message);
    }

    void blockStatement::printError(const std::string &message) const {
      source->printError(message);
    }
    //====================================
  }
}
