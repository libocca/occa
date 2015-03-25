#ifndef OCCA_PARSER_MAGIC_HEADER
#define OCCA_PARSER_MAGIC_HEADER

#include "occaParser.hpp"

namespace occa {
  namespace parserNS {
    class parserBase;

    class expInfo {
    public:
      bool isConstant;
      expNode exp;
    };

    class iteratorInfo {
    public:
      expInfo bounds[2], stride;
    };

    class viInfo {
    public:
      bool isAnIterator, isConstant, isUseless;
    };

    typedef std::map<varInfo*, viInfo*> viInfoMap_t_;
    typedef viInfoMap_t_::iterator     viInfoIterator;

    class viInfoMap_t {
    public:
      viInfoMap_t_ viMap;
      viInfo *anonVar; // Stores non-restrict variables

      viInfoMap_t();
      void free();

      void addVariable(varInfo &var);
    };

    class magician {
    public:
      parserBase &parser;
      statement &globalScope;
      varUsedMap_t &varUpdateMap;
      varUsedMap_t &varUsedMap;

      std::stack<viInfoMap_t> viInfoMapStack;

      static const bool analyzeEmbeddedStatements_f = true;

      magician(parserBase &parser_);

      viInfoMap_t* currentViInfoMap();
      void pushMapStack();
      void popMapStack();

      static void castMagicOn(parserBase &parser_);

      void castMagic();
      void analyzeFunction(statement &fs);
      void analyzeStatement(statement &s);

      void analyzeEmbeddedStatements(statement &s);

      void analyzeUpdateExpression(expNode &e, const int pos);
      bool analyzeForStatement(statement &s);
      bool analyzeWhileStatement(statement &s);
      void analyzeIfStatement(statementNode *snStart, statementNode *snEnd);
      void analyzeSwitchStatement(statement &s);

      bool statementGuaranteesBreak(statement &s);

      void addVariableWrite(expNode &varNode);
      void addVariableWrite(expNode &varNode,
                            const int brackets,
                            expNode *bracketNode);

      void addVariableRead(expNode &varNode);
      void addVariableRead(expNode &varNode,
                           const int brackets,
                           expNode *bracketNode);

      void addExpressionRead(expNode &e);
    };
  };
};

#endif
