#ifndef OCCA_PARSER_DEFINES_HEADER
#define OCCA_PARSER_DEFINES_HEADER

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include <stack>
#include <map>

#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

namespace occa {
  namespace parserNamespace {
    class opHolder;
    class typeHolder;

    class macroInfo;

    class statement;

    class typeInfo;
    class varInfo;
    class kernelInfo;

    class parserBase;

    template <class TM> class node;
    class strNode;
  };

  //---[ Info ]-----------------------------------
  typedef parserNamespace::node<parserNamespace::statement*> statementNode;
  typedef parserNamespace::node<parserNamespace::varInfo*>   varInfoNode;

  typedef std::map<std::string,int>  macroMap_t;
  typedef macroMap_t::iterator       macroMapIterator;
  typedef macroMap_t::const_iterator cMacroMapIterator;

  typedef std::map<std::string,int>        keywordTypeMap_t;
  typedef keywordTypeMap_t::iterator       keywordTypeMapIterator;
  typedef keywordTypeMap_t::const_iterator cKeywordTypeMapIterator;

  typedef std::map<parserNamespace::opHolder,int> opTypeMap_t;
  typedef opTypeMap_t::iterator                   opTypeMapIterator;
  typedef opTypeMap_t::const_iterator             cOpTypeMapIterator;

  typedef std::vector<parserNamespace::typeInfo*> anonymousTypeMap_t;

  typedef std::map<std::string,parserNamespace::typeInfo*> scopeTypeMap_t;
  typedef scopeTypeMap_t::iterator                        scopeTypeMapIterator;
  typedef scopeTypeMap_t::const_iterator                  cScopeTypeMapIterator;

  typedef std::map<std::string,parserNamespace::varInfo*> scopeVarMap_t;
  typedef scopeVarMap_t::iterator                         scopeVarMapIterator;
  typedef scopeVarMap_t::const_iterator                   cScopeVarMapIterator;

  typedef std::map<parserNamespace::varInfo*,
                   parserNamespace::statement*> varOriginMap_t;
  typedef varOriginMap_t::iterator              varOriginMapIterator;
  typedef varOriginMap_t::const_iterator        cVarOriginMapIterator;

  typedef std::map<parserNamespace::varInfo*, statementNode> varUsedMap_t;
  typedef varUsedMap_t::iterator                             varUsedMapIterator;
  typedef varUsedMap_t::const_iterator                       cVarUsedMapIterator;

  typedef std::map<std::string,parserNamespace::kernelInfo*> kernelInfoMap_t;
  typedef kernelInfoMap_t::iterator                          kernelInfoIterator;
  typedef kernelInfoMap_t::const_iterator                    cKernelInfoIterator;

  typedef std::map<parserNamespace::statement*,int> loopSection_t;
  typedef loopSection_t::iterator                   loopSectionIterator;

  typedef void (parserNamespace::parserBase::*applyToAllStatements_t)(parserNamespace::statement &s);

  typedef void (parserNamespace::parserBase::*applyToStatementsDefiningVar_t)(parserNamespace::varInfo &info,
                                                                              parserNamespace::statement &s);

  typedef void (parserNamespace::parserBase::*applyToStatementsUsingVar_t)(parserNamespace::varInfo &info,
                                                                           parserNamespace::statement &s);

  typedef bool (parserNamespace::parserBase::*findStatementWith_t)(parserNamespace::statement &s);

  namespace parserNamespace {
    extern keywordTypeMap_t keywordType;
    extern keywordTypeMap_t cKeywordType, fortranKeywordType;

    extern bool cKeywordsAreInitialized;
    extern bool fortranKeywordsAreInitialized;

    // By default, parsingC is true;
    static const bool parsingFortran = false;

    //   ---[ Delimeters ]---------
    static const char whitespace[]     = " \t\r\n\v\f\0";

    static const char cWordDelimeter[]  = " \t\r\n\v\f!\"#%&'()*+,-./:;<=>?[]^{|}~\0";
    static const char cWordDelimeter2[] = "!=##%=&&&=*=+++=-=--->.*.*../=::<<<===>=>>^=|=||\0";
    static const char cWordDelimeter3[] = "->*...<<=>>=\0";

    static const char fortranWordDelimeter[]  = " \t\r\n\v\f\"%'()*+,-./;<=>\0";
    static const char fortranWordDelimeter2[] = "**/=::<===>=\0";

    //   ---[ Keyword Types ]---
    static const int everythingType       = 0xFFFFFFFF;

    static const int emptyType            = 0;
    static const int descriptorType       = (15 << 0);
    static const int typedefType          = (1  << 0); // typedef
    static const int structType           = (1  << 1); // struct, class
    static const int specifierType        = (1  << 2); // void, char, short, int
    static const int qualifierType        = (1  << 3); // const, restrict, volatile

    static const int operatorType         = (0x1F << 4);
    static const int unitaryOperatorType  = (3    << 4);
    static const int lUnitaryOperatorType = (1    << 4);
    static const int rUnitaryOperatorType = (1    << 5);
    static const int binaryOperatorType   = (3    << 6);
    static const int assOperatorType      = (1    << 7); // hehe
    static const int ternaryOperatorType  = (1    << 8);

    static const int parentheses          = (1 <<  9);
    static const int brace                = (1 << 10);
    static const int bracket              = (1 << 11);

    static const int startSection         = (1 << 12);
    static const int endSection           = (1 << 13);

    static const int startParentheses     = (parentheses | startSection);
    static const int endParentheses       = (parentheses | endSection);

    static const int startBrace           = (brace | startSection);
    static const int endBrace             = (brace | endSection);

    static const int startBracket         = (bracket | startSection);
    static const int endBracket           = (bracket | endSection);

    static const int endStatement         = (1 << 14);

    static const int flowControlType      = (1 << 15);

    static const int presetValue          = (1 << 16);
    static const int unknownVariable      = (1 << 17);

    static const int specialKeywordType   = (1 << 18);

    static const int macroKeywordType     = (1 << 19);

    static const int apiKeywordType       = (7 << 20);
    static const int occaKeywordType      = (1 << 20);
    static const int cudaKeywordType      = (1 << 21);
    static const int openclKeywordType    = (1 << 22);

    //   ---[ Types ]-----------
    static const int isTypeDef = (1 << 0);
    static const int isVarInfo = (1 << 1);

    // static const int structType       = (1 << 0); (Defined on top)
    static const int noType              = (1 << 1);
    static const int boolType            = (1 << 2);
    static const int charType            = (1 << 3);
    static const int shortType           = (1 << 4);
    static const int intType             = (1 << 5);
    static const int longType            = (1 << 6);
    static const int floatType           = (1 << 7);
    static const int doubleType          = (1 << 8);

    static const int pointerType         = (7 << 10);
    static const int pointerTypeMask     = (7 << 10);
    static const int heapPointerType     = (1 << 10);
    static const int stackPointerType    = (1 << 11);
    static const int constPointerType    = (1 << 12);

    static const int referenceType       = (1 << 13);

    static const int variableType        = (1  << 14);
    static const int functionTypeMask    = (15 << 15);
    static const int functionType        = (1  << 15);
    static const int protoType           = (1  << 16); // =P
    static const int functionCallType    = (1  << 17);
    static const int functionPointerType = (1  << 18);
    static const int gotoType            = (1  << 19);

    static const int textureType         = (1 << 20);

    //   ---[ Type Def Info ]---
    static const int podTypeDef          = (1 << 21);

    static const int structTypeDef       = (1 << 22);
    static const int classTypeDef        = (1 << 23);
    static const int unionTypeDef        = (1 << 24);
    static const int enumTypeDef         = (1 << 25);

    static const int functionTypeDef     = (1 << 26);

    static const int templateTypeDef     = (1 << 27);

    //   ---[ Instructions ]----
    static const int doNothing           = (1 << 0);
    static const int ignoring            = (3 << 1);
    static const int ignoreUntilNextHash = (1 << 1);
    static const int ignoreUntilEnd      = (1 << 2);
    static const int readUntilNextHash   = (1 << 3);
    static const int doneIgnoring        = (1 << 4);
    static const int startHash           = (1 << 5);

    static const int keepMacro           = (1 << 6);

    static const int readingCode          = 0;
    static const int insideCommentBlock   = 1;
    static const int finishedCommentBlock = 2;

    //   ---[ Statements ]------
    static const int invalidStatementType   = (1 << 0);

    static const int simpleStatementType    = (7 << 1);
    static const int typedefStatementType   = (1 << 1);
    static const int declareStatementType   = (1 << 2);
    static const int updateStatementType    = (1 << 3);

    static const int flowStatementType      = (255 <<  4);
    static const int forStatementType       = (1   <<  4);
    static const int whileStatementType     = (1   <<  5);
    static const int doWhileStatementType   = (3   <<  5);
    static const int ifStatementType        = (1   <<  7);
    static const int elseIfStatementType    = (3   <<  7);
    static const int elseStatementType      = (5   <<  7);
    static const int switchStatementType    = (1   << 10);
    static const int gotoStatementType      = (1   << 11);

    static const int blankStatementType     = (1 << 12);

    static const int functionStatementType  = (3 << 13);
    static const int functionDefinitionType = (1 << 13);
    static const int functionPrototypeType  = (1 << 14);
    static const int blockStatementType     = (1 << 15);
    static const int structStatementType    = (1 << 16);

    static const int occaStatementType      = (1 << 17);

    static const int macroStatementType     = (1 << 18);
    static const int skipStatementType      = (1 << 19);

    //   ---[ OCCA Fors ]------
    static const int occaOuterForShift = 0;
    static const int occaInnerForShift = 3;

    static const int occaOuterForMask = 7;
    static const int occaInnerForMask = 56;

    static const int notAnOccaFor = 64;
    //==============================================
  };
};

#endif
