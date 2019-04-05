#ifndef OCCA_LANG_STATEMENTPEEKER_HEADER
#define OCCA_LANG_STATEMENTPEEKER_HEADER

#include <map>

#include <occa/lang/attribute.hpp>
#include <occa/lang/keyword.hpp>

namespace occa {
  namespace lang {
    class statementContext_t;
    class tokenContext_t;

    class statementPeeker_t {
      typedef std::map<int, int> keywordToStatementMap;

    private:
      tokenContext_t &tokenContext;
      statementContext_t &smntContext;
      const keywords_t &keywords;
      nameToAttributeMap &attributeMap;
      bool success;

      int lastPeek;
      int lastPeekPosition;
      keywordToStatementMap keywordPeek;

    public:
      statementPeeker_t(tokenContext_t &tokenContext_,
                        statementContext_t &smntContext_,
                        const keywords_t &keywords_,
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
