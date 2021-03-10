#ifndef OCCA_INTERNAL_LANG_PARSER_ATTRIBUTELOADER_HEADER
#define OCCA_INTERNAL_LANG_PARSER_ATTRIBUTELOADER_HEADER

#include <occa/internal/lang/attribute.hpp>
#include <occa/internal/lang/tokenContext.hpp>
#include <occa/internal/lang/variable.hpp>

namespace occa {
  namespace lang {
    class parser_t;
    class statementContext_t;
    class tokenContext_t;
    class vartype_t;

    class attributeLoader_t {
    private:
      tokenContext_t &tokenContext;
      statementContext_t &smntContext;
      parser_t &parser;
      nameToAttributeMap &attributeMap;
      bool success;

      attributeLoader_t(tokenContext_t &tokenContext_,
                        statementContext_t &smntContext_,
                        parser_t &parser_,
                        nameToAttributeMap &attributeMap_);

      bool loadAttributes(attributeTokenMap &attrs);

      void loadAttribute(attributeTokenMap &attrs);

      void setAttributeArgs(attributeToken_t &attr,
                            tokenRangeVector &argRanges);


      friend bool loadAttributes(tokenContext_t &tokenContext,
                                 statementContext_t &smntContext,
                                 parser_t &parser,
                                 nameToAttributeMap &attributeMap,
                                 attributeTokenMap &attrs);

      friend attribute_t* getAttribute(nameToAttributeMap &attributeMap,
                                       const std::string &name);
    };

    bool loadAttributes(tokenContext_t &tokenContext,
                        statementContext_t &smntContext,
                        parser_t &parser,
                        nameToAttributeMap &attributeMap,
                        attributeTokenMap &attrs);

    attribute_t* getAttribute(nameToAttributeMap &attributeMap,
                              const std::string &name);
  }
}

#endif
