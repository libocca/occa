#ifndef OCCA_LANG_PARSER_TYPELOADER_HEADER
#define OCCA_LANG_PARSER_TYPELOADER_HEADER

namespace occa {
  namespace lang {
    class tokenContext_t;
    class statementContext_t;
    class keywords_t;
    class vartype_t;

    class typeLoader_t {
    private:
      tokenContext_t &tokenContext;
      statementContext_t &smntContext;
      const keywords_t &keywords;
      bool success;

      typeLoader_t(tokenContext_t &tokenContext_,
                   statementContext_t &smntContext_,
                   const keywords_t &keywords_);

      bool loadBaseType(vartype_t &vartype);

      bool loadType(vartype_t &vartype);

      void loadVartypeQualifier(token_t *token,
                                const qualifier_t &qualifier,
                                vartype_t &vartype);

      void setVartypePointers(vartype_t &vartype);
      void setVartypePointer(vartype_t &vartype);

      void setVartypeReference(vartype_t &vartype);

      friend bool loadBaseType(tokenContext_t &tokenContext,
                               statementContext_t &smntContext,
                               const keywords_t &keywords,
                               vartype_t &vartype);

      friend bool loadType(tokenContext_t &tokenContext,
                           statementContext_t &smntContext,
                           const keywords_t &keywords,
                           vartype_t &vartype);

      friend bool isLoadingStruct(tokenContext_t &tokenContext,
                                  statementContext_t &smntContext,
                                  const keywords_t &keywords);
    };

    bool loadBaseType(tokenContext_t &tokenContext,
                      statementContext_t &smntContext,
                      const keywords_t &keywords,
                      vartype_t &vartype);

    bool loadType(tokenContext_t &tokenContext,
                  statementContext_t &smntContext,
                  const keywords_t &keywords,
                  vartype_t &vartype);

    bool isLoadingStruct(tokenContext_t &tokenContext,
                         statementContext_t &smntContext,
                         const keywords_t &keywords);
  }
}

#endif
