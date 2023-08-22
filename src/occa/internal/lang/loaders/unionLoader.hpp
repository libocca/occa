#ifndef OCCA_INTERNAL_LANG_PARSER_UNIONLOADER_HEADER
#define OCCA_INTERNAL_LANG_PARSER_UNIONLOADER_HEADER

#include <occa/internal/lang/tokenContext.hpp>
#include <occa/internal/lang/statementContext.hpp>

namespace occa {
  namespace lang {
    class union_t;
    class parser_t;

    class unionLoader_t {
     public:
      tokenContext_t &tokenContext;
      statementContext_t &smntContext;
      parser_t &parser;

      unionLoader_t(tokenContext_t &tokenContext_,
                     statementContext_t &smntContext_,
                     parser_t &parser_);

      bool loadUnion(union_t *&type);

      friend bool loadUnion(tokenContext_t &tokenContext,
                             statementContext_t &smntContext,
                             parser_t &parser,
                             union_t *&type);
    };

    bool loadUnion(tokenContext_t &tokenContext,
                    statementContext_t &smntContext,
                    parser_t &parser,
                    union_t *&type);
  }
}

#endif
