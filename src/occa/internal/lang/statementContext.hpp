#ifndef OCCA_INTERNAL_LANG_STATEMENTCONTEXT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENTCONTEXT_HEADER

#include <list>

#include <occa/internal/lang/statement/blockStatement.hpp>

namespace occa {
  namespace lang {
    typedef std::list<blockStatement*> blockStatementList;

    class statementContext_t {
    public:
      blockStatement &root;
      blockStatement *up;

    private:
      blockStatementList upStack;

    public:
      statementContext_t(blockStatement &root_);

      void pushUp(blockStatement &newUp);
      void popUp();

      void clear();
    };
  }
}

#endif
