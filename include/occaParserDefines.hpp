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

#if (OCCA_OS & (LINUX_OS | OSX_OS))
#  include <unistd.h>
#else
#  include <io.h>
#endif

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "occaDefines.hpp"

namespace occa {
  namespace parserNS {
    class opHolder;
    class typeHolder;

    class macroInfo;

    class expNode;
    class statement;

    class typeInfo;
    class varInfo;
    class kernelInfo;

    class parserBase;

    template <class TM> class node;
  };

  //---[ Info ]-----------------------------------
  typedef parserNS::node<parserNS::statement*>        statementNode;
  typedef parserNS::node<parserNS::varInfo*>          varInfoNode;

  typedef std::map<std::string,int>                   macroMap_t;
  typedef macroMap_t::iterator                        macroMapIterator;
  typedef macroMap_t::const_iterator                  cMacroMapIterator;

  typedef std::map<std::string,int>                   keywordTypeMap_t;
  typedef keywordTypeMap_t::iterator                  keywordTypeMapIterator;
  typedef keywordTypeMap_t::const_iterator            cKeywordTypeMapIterator;

  typedef std::map<parserNS::opHolder,int>            opTypeMap_t;
  typedef opTypeMap_t::iterator                       opTypeMapIterator;
  typedef opTypeMap_t::const_iterator                 cOpTypeMapIterator;

  typedef std::vector<parserNS::typeInfo*>            anonymousTypeMap_t;

  typedef std::map<std::string,parserNS::typeInfo*>   scopeTypeMap_t;
  typedef scopeTypeMap_t::iterator                    scopeTypeMapIterator;
  typedef scopeTypeMap_t::const_iterator              cScopeTypeMapIterator;

  typedef std::map<std::string,parserNS::varInfo*>    scopeVarMap_t;
  typedef scopeVarMap_t::iterator                     scopeVarMapIterator;
  typedef scopeVarMap_t::const_iterator               cScopeVarMapIterator;

  typedef std::map<parserNS::varInfo*, statementNode> varUsedMap_t;
  typedef varUsedMap_t::iterator                      varUsedMapIterator;
  typedef varUsedMap_t::const_iterator                cVarUsedMapIterator;

  typedef std::map<std::string,parserNS::kernelInfo*> kernelInfoMap_t;
  typedef kernelInfoMap_t::iterator                   kernelInfoIterator;
  typedef kernelInfoMap_t::const_iterator             cKernelInfoIterator;

  typedef std::map<std::string, std::string>          strToStrMap_t;
  typedef strToStrMap_t::iterator                     strToStrMapIterator;
  typedef strToStrMap_t::const_iterator               cStrToStrMapIterator;

  typedef std::map<parserNS::statement*,int>          statementIdMap_t;
  typedef statementIdMap_t::iterator                  statementIdMapIterator;

  typedef std::map<parserNS::varInfo*,int>            varInfoIdMap_t;
  typedef varInfoIdMap_t::iterator                    varInfoIdMapIterator;

  typedef std::vector<int>                            intVector_t;
  typedef std::vector<parserNS::statement*>           statementVector_t;
  typedef std::vector<parserNS::varInfo*>             varInfoVector_t;

  typedef std::map<int,bool>                          idDepMap_t;
  typedef idDepMap_t::iterator                        idDepMapIterator;

  typedef void (parserNS::parserBase::*applyToAllStatements_t)(parserNS::statement &s);

  typedef void (parserNS::parserBase::*applyToStatementsDefiningVar_t)(parserNS::varInfo &info,
                                                                       parserNS::statement &s);

  typedef void (parserNS::parserBase::*applyToStatementsUsingVar_t)(parserNS::varInfo &info,
                                                                    parserNS::statement &s);

  typedef bool (parserNS::parserBase::*findStatementWith_t)(parserNS::statement &s);

  namespace parserNS {
    extern keywordTypeMap_t keywordType;
    extern keywordTypeMap_t cKeywordType, fortranKeywordType;

    extern bool cKeywordsAreInitialized;
    extern bool fortranKeywordsAreInitialized;

    // By default, parsingC is true;
    static const bool parsingFortran = false;

    //   ---[ Delimiters ]---------
    static const char whitespace[]     = " \t\r\n\v\f\0";

    static const char cWordDelimiter[]  = " \t\r\n\v\f!\"#%&'()*+,-./:;<=>?[]^{|}~\0";
    static const char cWordDelimiter2[] = "!=###<#>%=&&&=*=+++=-=--->.*../=::<<<===>=>>^=|=||\0";
    static const char cWordDelimiter3[] = "->*...<<=>>=\0";

    static const char fortranWordDelimiter[]  = " \t\r\n\v\f\"#%'()*+,-./;<=>\0";
    static const char fortranWordDelimiter2[] = "**/=::<===>=\0";

    //   ---[ Types ]-----------
    static const int isTypeDef = (1 << 0);
    static const int isVarInfo = (1 << 1);

    static const int structType          = (1 << 0); // (Same as [ preExpType ])
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

    static const int structDef           = (1 << 22);
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
    static const int forceLineRemoval    = (1 << 7);

    static const int readingCode          = 0;
    static const int insideCommentBlock   = 1;
    static const int finishedCommentBlock = 2;

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
