#include <occa/lang/statement/statement.hpp>
#include <occa/lang/statement/blockStatement.hpp>
#include <occa/lang/token.hpp>

namespace occa {
  namespace lang {
    namespace statementType {
      const int none         = (1 << 0);
      const int all          = -1;

      const int empty        = (1 << 1);

      const int directive    = (1 << 2);
      const int pragma       = (1 << 3);
      const int comment      = (1 << 4);

      const int block        = (1 << 5);
      const int namespace_   = (1 << 6);

      const int function     = (1 << 7);
      const int functionDecl = (1 << 8);

      const int class_       = (1 << 9);
      const int classAccess  = (1 << 10);

      const int enum_        = (1 << 11);
      const int union_       = (1 << 12);

      const int expression   = (1 << 13);
      const int declaration  = (1 << 14);

      const int goto_        = (1 << 15);
      const int gotoLabel    = (1 << 16);

      const int if_          = (1 << 17);
      const int elif_        = (1 << 18);
      const int else_        = (1 << 19);
      const int for_         = (1 << 20);
      const int while_       = (1 << 21);
      const int switch_      = (1 << 22);
      const int case_        = (1 << 23);
      const int default_     = (1 << 24);
      const int continue_    = (1 << 25);
      const int break_       = (1 << 26);

      const int return_      = (1 << 27);

      const int attribute    = (1 << 28);

      const int blockStatements = (
        block        |
        elif_        |
        else_        |
        for_         |
        functionDecl |
        if_          |
        namespace_   |
        switch_      |
        while_
      );
    }

    statement_t::statement_t(blockStatement *up_,
                             const token_t *source_) :
      up(up_),
      source(token_t::clone(source_)),
      attributes() {}

    statement_t::statement_t(blockStatement *up_,
                             const statement_t &other) :
      up(up_),
      source(token_t::clone(other.source)),
      attributes() {}

    statement_t::~statement_t() {
      delete source;
    }

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

    void statement_t::swapSource(statement_t &other) {
      token_t *prevSource = source;
      source = other.source;
      other.source = prevSource;
    }

    bool statement_t::hasInScope(const std::string &name) {
      if (up) {
        return up->hasInScope(name);
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
      printer pout;
      pout << (*this);
      return pout.str();
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
      io::stdout << toString();
    }

    void statement_t::printWarning(const std::string &message) const {
      source->printWarning(message);
    }

    void statement_t::printError(const std::string &message) const {
      source->printError(message);
    }

    printer& operator << (printer &pout,
                          const statement_t &smnt) {
      smnt.print(pout);
      return pout;
    }
  }
}
