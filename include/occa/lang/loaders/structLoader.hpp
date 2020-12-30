#ifndef OCCA_INTERNAL_LANG_PARSER_STRUCTLOADER_HEADER
#define OCCA_INTERNAL_LANG_PARSER_STRUCTLOADER_HEADER

#include <occa/internal/lang/tokenContext.hpp>
#include <occa/internal/lang/statementContext.hpp>

namespace occa {
  namespace lang {
    class struct_t;
    class parser_t;

    class structLoader_t {
     public:
      tokenContext_t &tokenContext;
      statementContext_t &smntContext;
      parser_t &parser;

      structLoader_t(tokenContext_t &tokenContext_,
                     statementContext_t &smntContext_,
                     parser_t &parser_);

      bool loadStruct(struct_t *&type);

      friend bool loadStruct(tokenContext_t &tokenContext,
                             statementContext_t &smntContext,
                             parser_t &parser,
                             struct_t *&type);
    };

    bool loadStruct(tokenContext_t &tokenContext,
                    statementContext_t &smntContext,
                    parser_t &parser,
                    struct_t *&type);
  }
}

#endif
