#ifndef OCCA_INTERNAL_LANG_STATEMENTPEEKER_HEADER
#define OCCA_INTERNAL_LANG_STATEMENTPEEKER_HEADER

#include <map>

#include <occa/internal/lang/attribute.hpp>
#include <occa/internal/lang/keyword.hpp>

namespace occa {
  namespace lang {
    class statementContext_t;
    class tokenContext_t;
    class parser_t;

    class statementPeeker_t {
      typedef std::map<int, int> keywordToStatementMap;

    private:
      tokenContext_t &tokenContext;
      statementContext_t &smntContext;
      parser_t &parser;
      nameToAttributeMap &attributeMap;
      bool success;

      int lastPeek;
      int lastPeekPosition;
      keywordToStatementMap keywordPeek;

    public:
      statementPeeker_t(tokenContext_t &tokenContext_,
                        statementContext_t &smntContext_,
                        parser_t &parser_,
                        nameToAttributeMap &attributeMap_);

      void clear();

      bool peek(attributeTokenMap &attributes,
                int &statementType);
      int uncachedPeek();

      void setupPeek(attributeTokenMap &attributes);

      int peekIdentifier(const int tokenIndex);
      bool isGotoLabel(const int tokenIndex);

      int peekOperator(const int tokenIndex);
    };
  }
}

#endif
