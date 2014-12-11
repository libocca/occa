#include "occaParserDefines.hpp"

namespace occa {
  namespace parserNS {
    keywordTypeMap_t keywordType;
    keywordTypeMap_t cKeywordType, fortranKeywordType;

    bool usingCKeywords                = false;
    bool cKeywordsAreInitialized       = false;
    bool fortranKeywordsAreInitialized = false;
  };
};
