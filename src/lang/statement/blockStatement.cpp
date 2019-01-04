#include <occa/lang/statement/blockStatement.hpp>

#include <occa/lang/transforms/builtins/fillExprIdentifiers.hpp>
#include <occa/lang/transforms/builtins/replacer.hpp>

namespace occa {
  namespace lang {
    blockStatement::blockStatement(blockStatement *up_,
                                   token_t *source_) :
      statement_t(up_, source_) {}

    blockStatement::blockStatement(blockStatement *up_,
                                   const blockStatement &other) :
      statement_t(up_, other) {
      copyFrom(other);
    }

    blockStatement::~blockStatement() {
      clear();
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

    std::string blockStatement::statementName() const {
      return "block";
    }

    bool blockStatement::hasInScope(const std::string &name) {
      if (scope.has(name)) {
        return true;
      }
      return (up
              ? up->hasInScope(name)
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

    bool blockStatement::addToScope(type_t &type,
                                    const bool force) {
      return scope.add(type, force);
    }

    bool blockStatement::addToScope(function_t &func,
                                    const bool force) {
      return scope.add(func, force);
    }

    bool blockStatement::addToScope(variable_t &var,
                                    const bool force) {
      return scope.add(var, force);
    }

    void blockStatement::removeFromScope(const std::string &name,
                                         const bool deleteSource) {
      scope.remove(name, deleteSource);
    }

    bool blockStatement::hasDirectlyInScope(const std::string &name) {
      return scope.has(name);
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
          child.up = NULL;
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
      delete &child;
    }

    void blockStatement::swap(blockStatement &other) {
      swapSource(other);
      swapScope(other);
      swapChildren(other);
    }

    void blockStatement::swapScope(blockStatement &other) {
      scope.swap(other.scope);
    }

    void blockStatement::swapChildren(blockStatement &other) {
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
  }
}
