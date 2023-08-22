#ifndef OCCA_INTERNAL_LANG_PARSER_ENUMLOADER_HEADER
#define OCCA_INTERNAL_LANG_PARSER_ENUMLOADER_HEADER

#include <occa/internal/lang/enumerator.hpp>
#include <occa/internal/lang/tokenContext.hpp>
#include <occa/internal/lang/statementContext.hpp>

namespace occa {
  namespace lang {
    class enum_t;
    class parser_t;

    class enumLoader_t {
     public:
      tokenContext_t &tokenContext;
      statementContext_t &smntContext;
      parser_t &parser;

      enumLoader_t(tokenContext_t &tokenContext_,
                   statementContext_t &smntContext_,
                   parser_t &parser_);

      bool loadEnum(enum_t *&type);

      friend bool loadEnum(tokenContext_t &tokenContext,
                           statementContext_t &smntContext,
                           parser_t &parser,
                           enum_t *&type);
    };

    bool loadEnum(tokenContext_t &tokenContext,
                    statementContext_t &smntContext,
                    parser_t &parser,
                    enum_t *&type);
  }
}

#endif
