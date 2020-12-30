#include <occa/internal/lang/statementContext.hpp>

namespace occa {
  namespace lang {
    statementContext_t::statementContext_t(blockStatement &root_) :
      root(root_),
      up(&root) {}

    void statementContext_t::clear() {
      up = &root;
      upStack.clear();
    }

    void statementContext_t::pushUp(blockStatement &newUp) {
      upStack.push_back(up);
      up = &newUp;
    }

    void statementContext_t::popUp() {
      if (upStack.size()) {
        up = upStack.back();
        upStack.pop_back();
      } else {
        up = &root;
      }
    }
  }
}
