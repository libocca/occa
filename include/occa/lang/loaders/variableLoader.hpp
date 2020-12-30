#ifndef OCCA_INTERNAL_LANG_PARSER_VARIABLELOADER_HEADER
#define OCCA_INTERNAL_LANG_PARSER_VARIABLELOADER_HEADER

#include <occa/internal/lang/attribute.hpp>

namespace occa {
  namespace lang {
    class tokenContext_t;
    class statementContext_t;
    class parser_t;

    class variableLoader_t {
    private:
      tokenContext_t &tokenContext;
      statementContext_t &smntContext;
      parser_t &parser;
      nameToAttributeMap &attributeMap;
      bool success;

      variableLoader_t(tokenContext_t &tokenContext_,
                       statementContext_t &smntContext_,
                       parser_t &parser_,
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
                               parser_t &parser,
                               nameToAttributeMap &attributeMap,
                               variable_t &var);

      friend bool loadVariable(tokenContext_t &tokenContext,
                               statementContext_t &smntContext,
                               parser_t &parser,
                               nameToAttributeMap &attributeMap,
                               vartype_t &vartype,
                               variable_t &var);

      friend bool loadFunction(tokenContext_t &tokenContext,
                               statementContext_t &smntContext,
                               parser_t &parser,
                               nameToAttributeMap &attributeMap,
                               function_t &func);

      friend bool isLoadingVariable(tokenContext_t &tokenContext,
                                    statementContext_t &smntContext,
                                    parser_t &parser,
                                    nameToAttributeMap &attributeMap);

      friend bool isLoadingFunction(tokenContext_t &tokenContext,
                                    statementContext_t &smntContext,
                                    parser_t &parser,
                                    nameToAttributeMap &attributeMap);

      friend bool isLoadingFunctionPointer(tokenContext_t &tokenContext,
                                           statementContext_t &smntContext,
                                           parser_t &parser,
                                           nameToAttributeMap &attributeMap);
    };

    void getArgumentRanges(tokenContext_t &tokenContext,
                           tokenRangeVector &argRanges);

    bool loadVariable(tokenContext_t &tokenContext,
                      statementContext_t &smntContext,
                      parser_t &parser,
                      nameToAttributeMap &attributeMap,
                      variable_t &var);

    bool loadVariable(tokenContext_t &tokenContext,
                      statementContext_t &smntContext,
                      parser_t &parser,
                      nameToAttributeMap &attributeMap,
                      vartype_t &vartype,
                      variable_t &var);

    bool loadFunction(tokenContext_t &tokenContext,
                      statementContext_t &smntContext,
                      parser_t &parser,
                      nameToAttributeMap &attributeMap,
                      function_t &func);

    bool isLoadingVariable(tokenContext_t &tokenContext,
                           statementContext_t &smntContext,
                           parser_t &parser,
                           nameToAttributeMap &attributeMap);

    bool isLoadingFunction(tokenContext_t &tokenContext,
                           statementContext_t &smntContext,
                           parser_t &parser,
                           nameToAttributeMap &attributeMap);

    bool isLoadingFunctionPointer(tokenContext_t &tokenContext,
                                  statementContext_t &smntContext,
                                  parser_t &parser,
                                  nameToAttributeMap &attributeMap);
  }
}

#include "variableLoader.tpp"

#endif
