#ifndef OCCA_PARSER_MAGIC_HEADER
#define OCCA_PARSER_MAGIC_HEADER

#include "occaParser.hpp"

namespace occa {
  namespace parserNS {
    class parserBase;

    class strideInfo {
    public:
      bool isConstant;
      varInfo *varName;
      expNode stride;

      strideInfo();
    };

    class accessInfo {
    public:
      bool isUseful;
      std::vector<strideInfo> strides;

      accessInfo();

      void load(expNode &root);

      int dim();

      varInfo& var(const int pos);

      strideInfo& operator [] (const int pos);
      strideInfo& operator [] (const std::string &varName);
    };

    class ctInfo {
    public:
      bool hasConstValue;
      typeHolder constValue;

      expNode loopBounds[2];
      expNode loopStride;

      std::vector<accessInfo> reads;
      std::vector<accessInfo> writes;
    };

    typedef std::map<varInfo*, ctInfo> ctMap_t;
    typedef ctMap_t::iterator          ctMapIterator;
    typedef ctMap_t::const_iterator    cCtMapIterator;

    class magician {
    public:
      parserBase &parser;
      statement &globalScope;
      varUsedMap_t &varUpdateMap;
      varUsedMap_t &varUsedMap;

      ctMap_t ctMap;

      magician(parserBase &parser_);

      static void castMagicOn(parserBase &parser_);

      void castMagic();
      void castMagicOn(statement &kernel);
    };
  };
};

#endif