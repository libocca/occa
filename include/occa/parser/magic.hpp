/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#ifndef OCCA_PARSER_MAGIC_HEADER
#define OCCA_PARSER_MAGIC_HEADER

#include "occa/defines.hpp"
#include "occa/parser/parser.hpp"

namespace occa {
  namespace parserNS {
    class parserBase;
    class viInfo_t;
    class infoDB_t;

    typedef std::map<statement*, int> smntInfoMap_t;
    typedef smntInfoMap_t::iterator   smntInfoIterator;

    typedef std::map<varInfo*, viInfo_t*> viInfoMap_t_;
    typedef viInfoMap_t_::iterator        viInfoIterator;

    namespace viType {
      static const int isUseless        = (1 << 0);
      static const int isAVariable      = (1 << 1);
      static const int isAnIterator     = (1 << 2);
      static const int isConstant       = (1 << 3);
      static const int isComplex        = (1 << 4);

      std::string infoToStr(const int info);
    }

    namespace analyzeInfo {
      // Statement Info
      static const int isExecuted  = (1 << 0);
      static const int isIgnored   = (1 << 1);
      static const int schrodinger = (isExecuted | isIgnored); // hehe

      static const int hasLCD       = (1 << 2);

      // Return Info
      static const int didntChange = 0;
      static const int changed     = 1;

      // Bounds Info
      static const int LB = 0;
      static const int UB = 1;
      static const int S  = 2;
    }

    class atomInfo_t {
    public:
      infoDB_t *db;

      int info;
      typeHolder constValue;
      expNode *exp;
      varInfo *var;

      atomInfo_t(infoDB_t *db_ = NULL);
      atomInfo_t(const atomInfo_t &ai);

      atomInfo_t& operator = (const atomInfo_t &ai);

      bool operator == (const std::string &str);
      bool operator == (expNode &e);
      bool operator == (atomInfo_t &ai);
      bool operator != (atomInfo_t &ai);

      void setDB(infoDB_t *db_);

      void load(expNode &e);
      void load(varInfo &var_);
      void load(const std::string &s);

      bool expandValue();
      bool expandValue(expNode &e);
      bool expandValue(expNode *&expRoot, varInfo &v);

      void saveTo(expNode &e, const int leafPos = 0);

      bool isComplex();

      std::string getInfoStr();

      friend std::ostream& operator << (std::ostream &out, atomInfo_t &info);
    };

    class valueInfo_t {
    public:
      infoDB_t *db;

      int info;
      int indices;
      atomInfo_t value;
      atomInfo_t *vars, *strides;

      valueInfo_t(infoDB_t *db_ = NULL);
      valueInfo_t(expNode &e, infoDB_t *db_ = NULL);

      valueInfo_t(const valueInfo_t &vi);
      valueInfo_t& operator = (const valueInfo_t &vi);

      bool operator == (valueInfo_t &vi);
      bool operator != (valueInfo_t &vi);

      void setDB(infoDB_t *db_);

      void allocVS(const int count);

      bool isConstant();
      typeHolder constValue();

      bool isUseless();
      bool isComplex();

      void load(expNode &e);
      void load(varInfo &var);
      void load(const std::string &s);

      void loadVS(expNode &e, const int pos);

      int iteratorsIn(expNode &e);
      bool hasAnIterator(expNode &e);

      bool isAnIteratorExp(expNode &e);
      int iteratorExpsIn(expNode &e);
      bool hasAnIteratorExp(expNode &e);

      bool expandValues();
      void reEvaluateStrides();

      void sortIndices();
      static int qSortIndices(const void *a, const void *b);

      void mergeIndices();

      void saveTo(expNode &e, const int leafPos = 0);
      void saveIndexTo(const int index, expNode &e, const int leafPos = 0);

      void update(expNode &op, expNode &e);

      int hasStride(expNode &e);
      int hasStride(const std::string &str);
      int hasStride(atomInfo_t &stride);

      bool hasComplexStride();
      bool stridesConflict();
      // Assumption that (this and v) are not complex
      //   nor have conflicting strides
      bool conflictsWith(valueInfo_t &v);
      void setBoundInfo(typeHolder *&bounds, bool *&hasBounds);

      void insertOp(const std::string &op,
                    expNode &value_);
      void insertOp(const std::string &op,
                    const std::string &value_);

      varInfo& varValue();
      varInfo& var(const int pos);
      atomInfo_t& stride(const int pos);

      friend std::ostream& operator << (std::ostream &out, valueInfo_t &info);
    };

    class accessInfo_t {
    public:
      infoDB_t *db;
      statement *s;

      int dim;
      valueInfo_t value;
      valueInfo_t *dimIndices;

      accessInfo_t(infoDB_t *db_ = NULL);
      accessInfo_t(const accessInfo_t &ai);

      accessInfo_t& operator = (const accessInfo_t &ai);

      void setDB(infoDB_t *db_);

      void load(expNode &varNode);
      void load(const int brackets, expNode &bracketNode);

      bool hasComplexAccess();
      bool stridesConflict();
      bool conflictsWith(accessInfo_t &ai);

      friend std::ostream& operator << (std::ostream &out, accessInfo_t &info);
    };

    class iteratorInfo_t {
    public:
      infoDB_t *db;

      statement *s;
      valueInfo_t start, end, stride;

      iteratorInfo_t(infoDB_t *db_ = NULL);

      bool operator == (iteratorInfo_t &iter);
      bool operator != (iteratorInfo_t &iter);

      void setDB(infoDB_t *db_);

      friend std::ostream& operator << (std::ostream &out, iteratorInfo_t &info);
    };

    class viInfo_t {
    public:
      infoDB_t *db;
      varInfo *var;

      int info;
      valueInfo_t    valueInfo;
      iteratorInfo_t iteratorInfo;

      std::vector<accessInfo_t> reads, writes;
      std::vector<bool> writeSetsValue;

      static const int writeValue = (1 << 0);
      static const int readValue  = (1 << 1);

      viInfo_t(infoDB_t *db_ = NULL, varInfo *var_ = NULL);

      void setDB(infoDB_t *db_);

      bool hasBeenInitialized();

      void addWrite(const bool isUpdated, expNode &varNode);
      void addWrite(const bool isUpdated, const int brackets, expNode &bracketNode);

      void addRead(expNode &varNode);
      void addRead(const int brackets, expNode &bracketNode);

      void updateValue(expNode &opNode, expNode &setNode);

      void statementHasLCD(statement *sEnd);
      void sharedStatementHaveLCD(statement *a, statement *b);

      statement* lastSetStatement();

      void checkComplexity();

      void checkLastInput(accessInfo_t &ai, const int inputType);

      friend std::ostream& operator << (std::ostream &out, viInfo_t &info);
    };

    class viInfoMap_t {
    public:
      infoDB_t *db;

      viInfoMap_t_ viMap;
      viInfo_t *anonVar; // Stores non-restrict variables

      viInfoMap_t(infoDB_t *db_);

      void setDB(infoDB_t *db_);

      void free();

      void add(varInfo &var);
      viInfo_t* has(varInfo &var);

      viInfo_t& operator [] (varInfo &var);
    };

    class infoDB_t {
    public:
      bool locked;

      viInfoMap_t viInfoMap;
      smntInfoMap_t smntInfoMap;
      std::stack<int> smntInfoStack;

      infoDB_t();

      void lock();
      void unlock();
      bool isLocked();

      int& getSmntInfo();

      void add(varInfo &var);
      viInfo_t* has(varInfo &var);

      void enteringStatement(statement &s);
      void leavingStatement();

      viInfo_t& operator [] (varInfo &var);

      bool varIsAnIterator(varInfo &var);

      void statementsHaveLCD(statement *s);
      bool statementHasLCD(statement &s);
    };

    class magician {
    public:
      parserBase &parser;
      statement &globalScope;

      intVector_t testedTileSizes;

      infoDB_t db;

      magician(parserBase &parser_);

      viInfoMap_t* currentViInfoMap();
      void pushMapStack();
      void popMapStack();

      static void castMagicOn(parserBase &parser_);

      void castMagic();
      statementNode* analyzeFunction(statement &fs);
      void analyzeStatement(statement &s);

      void analyzeEmbeddedStatements(statement &s);

      void analyzeDeclareStatement(expNode &e);
      void analyzeDeclareExpression(expNode &e, const int pos);

      void analyzeUpdateStatement(expNode &e);
      void analyzeUpdateExpression(expNode &e, const int pos);

      void analyzeForStatement(statement &s);
      void analyzeFortranForStatement(statement &s);

      void analyzeWhileStatement(statement &s);

      void analyzeIfStatement(statementNode *snStart, statementNode *snEnd);

      void analyzeSwitchStatement(statement &s);

      bool statementGuaranteesBreak(statement &s);

      statementNode* generatePossibleKernels(statement &kernel);
      void storeInnerLoopCandidates(statementVector_t &loopsVec,
                                    intVector_t &depthVec,
                                    int outerLoopIdx,
                                    int innerLoopIdx,
                                    intVector_t &innerLoopVec);
      void storeNextDepthLoops(statementVector_t &loopsVec,
                               intVector_t &depthVec,
                               int loopIdx,
                               intVector_t &ndLoopsVec);
      bool nestedLoopHasSameBounds(statementVector_t &loopsVec,
                                   intVector_t &depthVec,
                                   int ndLoopIdx,
                                   iteratorInfo_t &iteratorInfo,
                                   intVector_t &innerLoopVec,
                                   const bool isFirstCall = true);
      statementNode* generateKernelsAndLabelLoops(statementVector_t &loopsVec,
                                                  intVector_t &depthVec,
                                                  intVector_t &outerLoopVec,
                                                  intVecVector_t &innerLoopVec);
      void storeLoopsAndDepths(statement &s,
                               statementVector_t &loopsVec,
                               intVector_t &depthVec, int depth);

      iteratorInfo_t iteratorLoopBounds(statement &s);
      void updateLoopBounds(statement &s);

      void printIfLoopsHaveLCD(statement &s);

      void addVariableWrite(expNode &varNode, expNode &opNode, expNode &setNode);
      void addVariableWrite(expNode &varNode,
                            expNode &opNode,
                            expNode &setNode,
                            const int brackets,
                            expNode &bracketNode);

      void addVariableRead(expNode &varNode);
      void addVariableRead(expNode &varNode,
                           const int brackets,
                           expNode &bracketNode);

      void addExpressionRead(expNode &e);

      //---[ Helper Functions ]---------
      static void placeAddedExps(infoDB_t &db, expNode &e, expVector_t &sumNodes);
      static void placeMultExps(infoDB_t &db, expNode &e, expVector_t &sumNodes);
      static void placeExps(infoDB_t &db, expNode &e, expVector_t &sumNodes, const std::string &delimiters);
      static bool expHasOp(expNode &e, const std::string &delimiters);

      static void simplify(infoDB_t &db, expNode &e);

      static void removePermutations(infoDB_t &db, expNode &e);
      static void turnMinusIntoNegatives(infoDB_t &db, expNode &e);

      static void mergeConstants(infoDB_t &db, expNode &e);
      static void mergeConstantsIn(infoDB_t &db, expNode &e);
      static void applyConstantsIn(infoDB_t &db, expNode &e);
      static void mergeVariables(infoDB_t &db, expNode &e);

      static void expandExp(infoDB_t &db, expNode &e);
      static void expandMult(infoDB_t &db, expNode &e);
      static void removeParentheses(infoDB_t &db, expNode &e);

      static expVector_t iteratorsIn(infoDB_t &db, expVector_t &v);
      static bool iteratorsMatch(expVector_t &a, expVector_t &b);
      static expVector_t removeItersFromExpVec(expVector_t &a, expVector_t &b);
      static void multiplyExpVec(expVector_t &v, expNode &e);
      static void sumExpVec(expVector_t &v, expNode &e);
      static void applyOpToExpVec(expVector_t &v, expNode &e, const std::string &op);
    };
  }
}

#endif
