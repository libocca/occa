#ifndef OCCA_PARSER_MAGIC_HEADER
#define OCCA_PARSER_MAGIC_HEADER

#include "occaParser.hpp"

namespace occa {
  namespace parserNS {
    class parserBase;
    class viInfo_t;

    typedef std::map<statement*, int> smntInfoMap_t;
    typedef smntInfoMap_t::iterator   smntInfoIterator;

    typedef std::map<varInfo*, viInfo_t*> viInfoMap_t_;
    typedef viInfoMap_t_::iterator        viInfoIterator;

    namespace viType {
      static const int isUseless        = (1 << 0);
      static const int isAVariable      = (1 << 1);
      static const int isAnIterator     = (1 << 2);
      static const int isConstant       = (1 << 3);
    };

    namespace analyzeInfo {
      static const int isExecuted      = (1 << 1);
      static const int isIgnored       = (1 << 2);
      static const int schrodinger     = (isExecuted | isIgnored); // hehe
    };

    class atomInfo_t {
    public:
      int info;
      typeHolder constValue;
      expNode exp;
      varInfo *var;

      atomInfo_t();

      void load(expNode &e);
      void load(varInfo &var_);
      void load(const std::string &s);
    };

    class valueInfo_t {
    public:
      int info;
      int indices;
      atomInfo_t value;
      atomInfo_t *vars, *strides;

      valueInfo_t();
      valueInfo_t(const valueInfo_t &vi);
      valueInfo_t(expNode &e);

      valueInfo_t& operator = (const valueInfo_t &vi);

      void allocVS(const int count);

      bool isUseless();

      void load(expNode &e);
      void load(varInfo &var);
      void load(const std::string &s);

      void loadVS(expNode &e, const int pos);

      void merge(expNode &op, expNode &e);

      typeHolder constValue();
      varInfo& varValue();
      varInfo& var(const int pos);
      atomInfo_t& stride(const int pos);
    };

    class accessInfo_t {
    public:
      statement *s;

      int dim;
      valueInfo_t value;
      valueInfo_t *dimIndices;

      accessInfo_t();

      void load(expNode &varNode);
      void load(const int brackets, expNode &bracketNode);

      bool conflictsWith(accessInfo_t &ai);
    };

    class iteratorInfo_t {
    public:
      valueInfo_t start, end, stride;

      iteratorInfo_t();
    };

    class viInfo_t {
    public:
      int info;
      valueInfo_t    valueInfo;
      accessInfo_t   dimInfo;
      iteratorInfo_t iteratorInfo;

      std::vector<accessInfo_t> reads, writes;

      static const int writeValue = (1 << 0);
      static const int readValue  = (1 << 1);

      viInfo_t();

      accessInfo_t& addWrite(expNode &varNode);
      accessInfo_t& addWrite(const int brackets, expNode &bracketNode);

      accessInfo_t& addRead(expNode &varNode);
      accessInfo_t& addRead(const int brackets, expNode &bracketNode);

      void checkLastInput(accessInfo_t &ai, const int inputType);
    };

    class viInfoMap_t {
    public:
      viInfoMap_t_ viMap;
      viInfo_t *anonVar; // Stores non-restrict variables

      viInfoMap_t();

      void free();

      void add(varInfo &var);
      viInfo_t* has(varInfo &var);

      viInfo_t& operator [] (varInfo &var);
    };

    class viInfoDB_t {
    public:
      viInfoMap_t viInfoMap;
      smntInfoMap_t smntInfoMap;

      void add(varInfo &var);
      viInfo_t* has(varInfo &var);

      void enteringStatement(statement &s);
      void leavingStatement();

      viInfo_t& operator [] (varInfo &var);
    };

    class magician {
    public:
      parserBase &parser;
      statement &globalScope;
      varUsedMap_t &varUpdateMap;
      varUsedMap_t &varUsedMap;

      viInfoDB_t viInfoDB;

      magician(parserBase &parser_);

      viInfoMap_t* currentViInfoMap();
      void pushMapStack();
      void popMapStack();

      static void castMagicOn(parserBase &parser_);

      void castMagic();
      void analyzeFunction(statement &fs);
      void analyzeStatement(statement &s);

      void analyzeEmbeddedStatements(statement &s);

      void analyzeDeclareStatement(int &smntInfo, expNode &e);
      void analyzeDeclareExpression(int &smntInfo, expNode &e, const int pos);

      void analyzeUpdateStatement(int &smntInfo, expNode &e);
      void analyzeUpdateExpression(int &smntInfo, expNode &e, const int pos);

      void analyzeForStatement(int &smntInfo, statement &s);
      void analyzeFortranForStatement(int &smntInfo, statement &s);

      void analyzeWhileStatement(int &smntInfo, statement &s);

      void analyzeIfStatement(int &smntInfo, statementNode *snStart, statementNode *snEnd);

      void analyzeSwitchStatement(int &smntInfo, statement &s);

      bool statementGuaranteesBreak(statement &s);

      bool variableIsUpdated(expNode &varNode);

      void addVariableWrite(expNode &varNode, expNode &setNode);
      void addVariableWrite(expNode &varNode,
                            expNode &setNode,
                            const int brackets,
                            expNode &bracketNode);

      void addVariableRead(expNode &varNode);
      void addVariableRead(expNode &varNode,
                           const int brackets,
                           expNode &bracketNode);

      void addExpressionRead(expNode &e);
    };
  };
};

#endif
