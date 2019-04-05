#ifndef OCCA_LANG_PARSER_VARIABLELOADER_HEADER
#define OCCA_LANG_PARSER_VARIABLELOADER_HEADER

#include <occa/lang/attribute.hpp>

namespace occa {
  namespace lang {
    class tokenContext_t;
    class statementContext_t;
    class keywords_t;

    class variableLoader_t {
    private:
      tokenContext_t &tokenContext;
      statementContext_t &smntContext;
      const keywords_t &keywords;
      nameToAttributeMap &attributeMap;
      bool success;

      variableLoader_t(tokenContext_t &tokenContext_,
                       statementContext_t &smntContext_,
                       const keywords_t &keywords_,
                       nameToAttributeMap &attributeMap_);

      bool loadVariable(variable_t &var);

      bool loadVariable(vartype_t &vartype,
                        variable_t &var);

      bool isLoadingVariable();
      bool isLoadingFunction();
      bool isLoadingFunctionPointer();

      bool loadBasicVariable(vartype_t &vartype,
                             variable_t &var);

      bool loadFunctionPointer(vartype_t &vartype,
                               variable_t &functionVar);

      bool loadFunction(function_t &func);

      bool hasArray();
      void setArrays(vartype_t &vartype);

      void setArguments(functionPtr_t &func);
      void setArguments(function_t &func);

      template <class funcType>
      void setArgumentsFor(funcType &func);

      friend bool loadVariable(tokenContext_t &tokenContext,
                               statementContext_t &smntContext,
                               const keywords_t &keywords,
                               nameToAttributeMap &attributeMap,
                               variable_t &var);

      friend bool loadVariable(tokenContext_t &tokenContext,
                               statementContext_t &smntContext,
                               const keywords_t &keywords,
                               nameToAttributeMap &attributeMap,
                               vartype_t &vartype,
                               variable_t &var);

      friend bool loadFunction(tokenContext_t &tokenContext,
                               statementContext_t &smntContext,
                               const keywords_t &keywords,
                               nameToAttributeMap &attributeMap,
                               function_t &func);

      friend bool isLoadingVariable(tokenContext_t &tokenContext,
                                    statementContext_t &smntContext,
                                    const keywords_t &keywords,
                                    nameToAttributeMap &attributeMap);

      friend bool isLoadingFunction(tokenContext_t &tokenContext,
                                    statementContext_t &smntContext,
                                    const keywords_t &keywords,
                                    nameToAttributeMap &attributeMap);

      friend bool isLoadingFunctionPointer(tokenContext_t &tokenContext,
                                           statementContext_t &smntContext,
                                           const keywords_t &keywords,
                                           nameToAttributeMap &attributeMap);
    };

    void getArgumentRanges(tokenContext_t &tokenContext,
                           tokenRangeVector &argRanges);

    bool loadVariable(tokenContext_t &tokenContext,
                      statementContext_t &smntContext,
                      const keywords_t &keywords,
                      nameToAttributeMap &attributeMap,
                      variable_t &var);

    bool loadVariable(tokenContext_t &tokenContext,
                      statementContext_t &smntContext,
                      const keywords_t &keywords,
                      nameToAttributeMap &attributeMap,
                      vartype_t &vartype,
                      variable_t &var);

    bool loadFunction(tokenContext_t &tokenContext,
                      statementContext_t &smntContext,
                      const keywords_t &keywords,
                      nameToAttributeMap &attributeMap,
                      function_t &func);

    bool isLoadingVariable(tokenContext_t &tokenContext,
                           statementContext_t &smntContext,
                           const keywords_t &keywords,
                           nameToAttributeMap &attributeMap);

    bool isLoadingFunction(tokenContext_t &tokenContext,
                           statementContext_t &smntContext,
                           const keywords_t &keywords,
                           nameToAttributeMap &attributeMap);

    bool isLoadingFunctionPointer(tokenContext_t &tokenContext,
                                  statementContext_t &smntContext,
                                  const keywords_t &keywords,
                                  nameToAttributeMap &attributeMap);
  }
}

#include "variableLoader.tpp"

#endif
