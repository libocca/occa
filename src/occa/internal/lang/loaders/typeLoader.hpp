#ifndef OCCA_INTERNAL_LANG_PARSER_TYPELOADER_HEADER
#define OCCA_INTERNAL_LANG_PARSER_TYPELOADER_HEADER

namespace occa {
  namespace lang {
    class tokenContext_t;
    class statementContext_t;
    class parser_t;
    class vartype_t;

    class typeLoader_t {
    private:
      tokenContext_t &tokenContext;
      statementContext_t &smntContext;
      parser_t &parser;
      bool success;

      typeLoader_t(tokenContext_t &tokenContext_,
                   statementContext_t &smntContext_,
                   parser_t &parser_);

      bool loadBaseType(vartype_t &vartype);

      bool loadType(vartype_t &vartype);

      void loadVartypeQualifier(token_t *token,
                                const qualifier_t &qualifier,
                                vartype_t &vartype);

      void setVartypePointers(vartype_t &vartype);
      void setVartypePointer(vartype_t &vartype);

      void setVartypeReference(vartype_t &vartype);

      void loadStruct(vartype_t &vartype);

      friend bool loadType(tokenContext_t &tokenContext,
                           statementContext_t &smntContext,
                           parser_t &parser,
                           vartype_t &vartype);

      friend bool loadBaseType(tokenContext_t &tokenContext,
                               statementContext_t &smntContext,
                               parser_t &parser,
                               vartype_t &vartype);

      friend bool isLoadingStruct(tokenContext_t &tokenContext,
                                  statementContext_t &smntContext,
                                  parser_t &parser);
    };

    bool loadType(tokenContext_t &tokenContext,
                  statementContext_t &smntContext,
                  parser_t &parser,
                  vartype_t &vartype);

    bool loadBaseType(tokenContext_t &tokenContext,
                      statementContext_t &smntContext,
                      parser_t &parser,
                      vartype_t &vartype);

    bool isLoadingStruct(tokenContext_t &tokenContext,
                         statementContext_t &smntContext,
                         parser_t &parser);
  }
}

#endif
