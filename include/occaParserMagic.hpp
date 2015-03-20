#ifndef OCCA_PARSER_MAGIC_HEADER
#define OCCA_PARSER_MAGIC_HEADER

#include "occaParser.hpp"

namespace occa {
  namespace parserNS {
    class parserBase;

    class magician {
    public:
      parserBase &parser;
      statement &globalScope;
      varUsedMap_t &varUpdateMap;
      varUsedMap_t &varUsedMap;

      magician(parserBase &parser_);

      static void castMagicOn(parserBase &parser_);

      void castMagic();
      void castMagicOn(statement &kernel);
    };
  };
};

#endif