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

      magician(parserBase &parser_);

      void castAutomatic();
      void castAutomaticOn(statement &kernel);
    };
  };
};

#endif

/*
  1: x = 1;
  2: y = x + 2;
  3: x = z - w;
  4: x = y / z;

  Flow:
    1 -> 2: y uses x
    2 -> 4: x uses y

  Anti:
    2 -> 3: x needs to wait for y to use x

  Out :
    1 -> 3: x is updated
    3 -> 4: x is updated

  In  :
    3 -> 4: z is used






 */