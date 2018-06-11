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
#include <occa/lang/baseStatement.hpp>
#include <occa/lang/token.hpp>
#include <occa/lang/type.hpp>
#include <occa/lang/builtins/transforms/fillExprIdentifiers.hpp>
#include <occa/lang/builtins/transforms/replacer.hpp>

namespace occa {
  namespace lang {
    namespace statementType {
      const int none         = (1 << 0);
      const int all          = -1;

      const int empty        = (1 << 1);

      const int directive    = (1 << 2);
      const int pragma       = (1 << 3);

      const int block        = (1 << 4);
      const int namespace_   = (1 << 5);

      const int typeDecl     = (1 << 6);
      const int function     = (1 << 7);
      const int functionDecl = (1 << 8);
      const int classAccess  = (1 << 9);

      const int expression   = (1 << 10);
      const int declaration  = (1 << 11);

      const int goto_        = (1 << 12);
      const int gotoLabel    = (1 << 13);

      const int if_          = (1 << 14);
      const int elif_        = (1 << 15);
      const int else_        = (1 << 16);
      const int for_         = (1 << 17);
      const int while_       = (1 << 18);
      const int switch_      = (1 << 19);
      const int case_        = (1 << 20);
      const int default_     = (1 << 21);
      const int continue_    = (1 << 22);
      const int break_       = (1 << 23);

      const int return_      = (1 << 24);

      const int attribute    = (1 << 25);
    }

    statement_t::statement_t(blockStatement *up_) :
      up(up_),
      attributes() {}

    statement_t::~statement_t() {}

    statement_t& statement_t::clone(blockStatement *up_) const {
      statement_t &s = clone_(up_);
      s.attributes = attributes;
      return s;
    }

    statement_t* statement_t::clone(blockStatement *up_,
                                    statement_t *smnt) {
      if (smnt) {
        return &(smnt->clone(up_));
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

    void statement_t::removeFromParent() {
      if (up) {
        up->remove(*this);
      }
    }

    void statement_t::debugPrint() const {
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

    statement_t& emptyStatement::clone_(blockStatement *up_) const {
      return *(new emptyStatement(up_,
                                  source));
    }

    int emptyStatement::type() const {
      return statementType::empty;
    }

    void emptyStatement::print(printer &pout) const {
      if (hasSemicolon) {
        pout.printStartIndentation();
        pout << ';';
        pout.printEndNewline();
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

    blockStatement::blockStatement(blockStatement *up_,
                                   const blockStatement &other) :
      statement_t(up_),
      source(token_t::clone(other.source)) {
      copyFrom(other);
    }

    blockStatement::~blockStatement() {
      clear();
      delete source;
    }

    statement_t& blockStatement::clone_(blockStatement *up_) const {
      return *(new blockStatement(up_, *this));
    }

    // Block statements such as for/if/while/etc need to replace
    //   variables after their inner-children statements are set
    void blockStatement::copyFrom(const blockStatement &other) {
      attributes = other.attributes;

      // Copy children
      const int childCount = (int) other.children.size();
      for (int i = 0; i < childCount; ++i) {
        add(other.children[i]->clone(this));
      }

      // Replace keywords
      keywordMap &keywords = scope.keywords;
      const keywordMap &otherKeywords = other.scope.keywords;

      keywordMap::iterator it = keywords.begin();
      while (it != keywords.end()) {
        const std::string &name = it->first;
        keyword_t &keyword = *(it->second);

        keywordMap::const_iterator oit = otherKeywords.find(name);
        if (oit != otherKeywords.end()) {
          replaceKeywords(*this,
                          *(oit->second),
                          keyword);
        }
        ++it;
      }
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

    void blockStatement::remove(statement_t &child) {
      const int childCount = (int) children.size();
      for (int i = 0; i < childCount; ++i) {
        if (children[i] == &child) {
          children.erase(children.begin() + i);
          return;
        }
      }
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
