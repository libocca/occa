#ifndef OCCA_PARSER_HEADER
#define OCCA_PARSER_HEADER

#include <iostream>
#include <sstream>
#include <vector>
#include <stack>
#include <map>

#include <cstring>
#include <cstdlib>
#include <cstdio>

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
    class varInfo;

    class parserBase;

    template <class TM> class node;
    class strNode;

    //---[ Info ]-----------------------------------
    typedef node<statement*> statementNode;
    typedef node<varInfo*>   varInfoNode;

    typedef std::map<std::string,int> macroMap_t;
    typedef macroMap_t::iterator      macroMapIterator;

    typedef std::map<std::string,int>  keywordTypeMap_t;
    typedef keywordTypeMap_t::iterator keywordTypeMapIterator;

    typedef std::map<opHolder,int> opTypeMap_t;
    typedef opTypeMap_t::iterator  opTypeMapIterator;

    typedef std::map<std::string,varInfo*> scopeVarMap_t;
    typedef scopeVarMap_t::iterator        scopeVarMapIterator;

    typedef std::map<varInfo*,statement*> varOriginMap_t;
    typedef varOriginMap_t::iterator      varOriginMapIterator;

    typedef std::map<varInfo*,statementNode> varUsedMap_t;
    typedef varUsedMap_t::iterator           varUsedMapIterator;

    typedef std::map<statement*,int> loopSection_t;
    typedef loopSection_t::iterator  loopSectionIterator;

    typedef void (parserBase::*applyToAllStatements_t)(statement &s);
    typedef void (parserBase::*applyToStatementsDefiningVar_t)(varInfo &info, statement &s);
    typedef void (parserBase::*applyToStatementsUsingVar_t)(varInfo &info, statement &s);

    typedef bool (parserBase::*findStatementWith_t)(statement &s);

    keywordTypeMap_t keywordType;
    opTypeMap_t opPrecedence;

    bool keywordsAreInitialized = false;

    //   ---[ Delimeters ]---------
    static const char whitespace[]       = " \t\r\n\v\f\0";
    static const char wordDelimeter[]    = " \t\r\n\v\f!\"#%&'()*+,-./:;<=>?[]^{|}~\0";
    static const char wordDelimeterExt[] = "!=##%=&&&=*=+++=-=--->/=::<<<===>=>>^=|=||\0";

    //   ---[ Keyword Types ]---
    static const int everythingType       = 0xFFFFFFFF;

    static const int descriptorType       = (7 << 0);
    static const int specifierType        = (1 << 0); // void, char, short, int
    static const int qualifierType        = (1 << 1); // const, restrict, volatile
    static const int structType           = (1 << 2); // struct, class, typedef

    static const int operatorType         = (0x1F << 3);
    static const int unitaryOperatorType  = (3    << 3);
    static const int lUnitaryOperatorType = (1    << 3);
    static const int rUnitaryOperatorType = (1    << 4);
    static const int binaryOperatorType   = (3    << 5);
    static const int assOperatorType      = (1    << 6); // hehe
    static const int ternaryOperatorType  = (1    << 7);

    static const int nameType             = (3 << 8);
    static const int variableNameType     = (1 << 8);
    static const int functionNameType     = (1 << 9);

    static const int parentheses          = (1 << 10);
    static const int brace                = (1 << 11);
    static const int bracket              = (1 << 12);

    static const int startSection         = (1 << 13);
    static const int endSection           = (1 << 14);

    static const int startParentheses     = (parentheses | startSection);
    static const int endParentheses       = (parentheses | endSection);

    static const int startBrace           = (brace | startSection);
    static const int endBrace             = (brace | endSection);

    static const int startBracket         = (bracket | startSection);
    static const int endBracket           = (bracket | endSection);

    static const int endStatement         = (1 << 15);

    static const int flowControlType      = (1 << 16);

    static const int presetValue          = (1 << 17);
    static const int unknownVariable      = (1 << 18);

    static const int specialKeywordType   = (1 << 19);

    static const int apiKeywordType       = (7 << 21);
    static const int occaKeywordType      = (1 << 21);
    static const int cudaKeywordType      = (1 << 22);
    static const int openclKeywordType    = (1 << 23);

    //   ---[ Types ]-----------
    static const int noType           = (1 << 1);
    static const int boolType         = (1 << 2);
    static const int charType         = (1 << 3);
    static const int shortType        = (1 << 4);
    static const int intType          = (1 << 5);
    static const int longType         = (1 << 6);
    static const int floatType        = (1 << 7);
    static const int doubleType       = (1 << 8);

    static const int pointerType      = (3 << 9);
    static const int heapPointerType  = (1 << 9);
    static const int stackPointerType = (1 << 10);
    static const int constPointerType = (1 << 11);

    static const int referenceType    = (1 << 12);

    static const int variableType     = (1 << 13);
    static const int functionType     = (1 << 14);
    static const int protoType        = (1 << 15); // =P
    static const int functionCallType = (1 << 16);
    static const int gotoType         = (1 << 17);

    static const int textureType      = (1 << 18);

    //   ---[ Instructions ]----
    static const int doNothing           = (1 << 0);
    static const int ignoring            = (3 << 1);
    static const int ignoreUntilNextHash = (1 << 1);
    static const int ignoreUntilEnd      = (1 << 2);
    static const int readUntilNextHash   = (1 << 3);
    static const int doneIgnoring        = (1 << 4);
    static const int startHash           = (1 << 5);

    static const int readingCode          = 0;
    static const int insideCommentBlock   = 1;
    static const int finishedCommentBlock = 2;

    //   ---[ Statements ]------
    static const int simpleStatementType    = (3 << 0);
    static const int declareStatementType   = (1 << 0);
    static const int updateStatementType    = (1 << 1);

    static const int flowStatementType      = (255 << 2);
    static const int forStatementType       = (1   << 2);
    static const int whileStatementType     = (1   << 3);
    static const int doWhileStatementType   = (3   << 3);
    static const int ifStatementType        = (1   << 5);
    static const int elseIfStatementType    = (3   << 5);
    static const int elseStatementType      = (5   << 5);
    static const int switchStatementType    = (1   << 8);
    static const int gotoStatementType      = (1   << 9);

    static const int blankStatementType     = (1 << 10);

    static const int functionStatementType  = (3 << 11);
    static const int functionDefinitionType = (1 << 11);
    static const int functionPrototypeType  = (1 << 12);
    static const int blockStatementType     = (1 << 13);
    static const int structStatementType    = (1 << 14);

    static const int occaStatementType      = (1 << 15);

    //   ---[ OCCA Fors ]------
    static const int occaOuterForShift = 0;
    static const int occaInnerForShift = 3;

    static const int occaOuterForMask = 7;
    static const int occaInnerForMask = 56;

    static const int notAnOccaFor = 64;
    //==============================================

    strNode* splitFileContents(const char *cRoot);
    strNode* labelCode(strNode *lineNodeRoot);

    class parserBase {
    public:
      macroMap_t macroMap;
      std::vector<macroInfo> macros;

      bool macrosAreInitialized;

      varOriginMap_t varOriginMap;
      varUsedMap_t varUsedMap;     // Statements are placed backwards

      inline parserBase(){};

      const std::string parseSource(const char *cRoot);
      const std::string parseFile(const std::string &filename);

      //---[ Macro Parser Functions ]-------
      std::string getMacroName(const char *&c);

      bool evaluateMacroStatement(const char *&c);
      typeHolder evaluateLabelNode(strNode *labelNodeRoot);

      void loadMacroInfo(macroInfo &info, const char *&c);
      int loadMacro(const std::string &line, const int state = doNothing);

      void applyMacros(std::string &line);

      strNode* preprocessMacros(strNode *nodeRoot);
      //====================================

      void initMacros();

      void applyToAllStatements(statement &s,
                                applyToAllStatements_t func);

      void applyToStatementsDefiningVar(applyToStatementsDefiningVar_t func);

      void applyToStatementsUsingVar(varInfo &info,
                                     applyToStatementsUsingVar_t func);

      // <<>>

      bool statementIsAKernel(statement &s);

      statement* getStatementKernel(statement &s);

      bool statementKernelUsesNativeOCCA(statement &s);

      bool statementKernelUsesNativeOKL(statement &s);

      bool statementKernelUsesNativeLanguage(statement &s);

      void addOccaForCounter(statement &s,
                             const std::string &ioLoop,
                             const std::string &loopNest);

      bool nodeHasUnknownVariable(strNode *n);

      void setupOccaFors(statement &s);

      void loadScopeVarMap(statement &s);

      bool statementHasOccaFor(statement &s);

      bool statementHasOklFor(statement &s);

      void labelKernelsAsNativeOrNot(statement &s);

      void setupCudaVariables(statement &s);

      void loadVariableInformation(statement &s,
                                   strNode *n);

      void loadVariableInformation(statement &s);

      void addFunctionPrototypes(statement &s);

      int statementOccaForNest(statement &s);
      bool statementIsAnOccaFor(statement &s);

      void fixOccaForStatementOrder(statement &origin, statementNode *sn);
      void fixOccaForOrder(statement &s);

      void addParallelFors(statement &s);

      void updateConstToConstant(statement &s);

      strNode* occaExclusiveStrNode(varInfo &info,
                                    const int depth,
                                    const int sideDepth);

      void addKernelInfo(varInfo &info, statement &s);

      void addArgQualifiers(varInfo &info, statement &s);

      void modifyExclusiveVariables(statement &s);

      void modifyStatementOccaForVariables(varInfo &var,
                                           strNode *n);

      void modifyOccaForVariables();

      void modifyTextureVariables();

      std::string occaScope(statement &s);

      void incrementDepth(statement &s);

      void decrementDepth(statement &s);

      bool statementHasBarrier(statement &s);

      statementNode* findStatementWith(statement &s,
                                       findStatementWith_t func);

      int getKernelOuterDim(statement &s);

      int getKernelInnerDim(statement &s);

      void checkPathForConditionals(statementNode *path);

      int findLoopSections(statement &s,
                           statementNode *path,
                           loopSection_t &loopSection,
                           int section = 0);

      bool varInTwoSegments(varInfo &info,
                            loopSection_t &loopSection);

      varInfoNode* findVarsMovingToTop(statement &s,
                                       loopSection_t &loopSection);

      void splitDefineForVariable(statement *&origin,
                                  varInfo &var, strNode *varNode,
                                  const int declPos);

      void addInnerForsBetweenBarriers(statement &origin,
                                       statementNode *s,
                                       const int innerDim);

      void addInnerFors(statement &s);

      void addOuterFors(statement &s);

      void addOccaForsToKernel(statement &s);

      void addOccaFors(statement &globalScope);

      void setupOccaVariables(statement &s);
      // <<>>
    };


    //---[ Op(erator) Holder ]----------------------
    class opHolder {
    public:
      std::string op;
      int type;

      opHolder(const std::string &op_, const int type_) :
        op(op_),
        type(type_) {}

      bool operator < (const opHolder &h) const {
        if(op < h.op)
          return true;
        else if(op > h.op)
          return false;
        else if(type < h.type)
          return true;

        return false;
      }
    };
    //==============================================


    //---[ Type Holder ]----------------------------
    bool isAnInt(const char *c);
    bool isAFloat(const char *c);

    int toInt(const char *c){
      return atoi(c);
    }

    bool toBool(const char *c){
      if(isAnInt(c))
        return (atoi(c) != 0);
      else if(isAFloat(c))
        return (atof(c) != 0);
      else if(strcmp(c, "true") == 0)
        return true;
      else if(strcmp(c, "false") == 0)
        return false;

      std::cout << "[" << c << "] is not a bool.\n";
      throw 1;

      return false;
    }

    char toChar(const char *c){
      return (char) atoi(c);
    }

    long toLong(const char *c){
      return atol(c);
    }

    short toShort(const char *c){
      return (short) atoi(c);
    }

    float toFloat(const char *c){
      return atof(c);
    }

    double toDouble(const char *c){
      return (double) atof(c);
    }

    std::string typeInfoToStr(const int typeInfo){
      std::string ret = "";

      if(typeInfo & heapPointerType){
        if(typeInfo & functionType)
          ret += "* ";
        else
          ret += " *";

        if(typeInfo & constPointerType)
          ret += " const ";
      }
      else if(typeInfo & referenceType){
        ret += " &";
      }
      else
        ret += " ";

      return ret;
    }

    class typeHolder {
    public:
      union {
        int int_;
        bool bool_;
        char char_;
        long long_;
        short short_;
        float float_;
        double double_;
      } value;

      int type;

      typeHolder(){
        value.double_ = 0;
        type = noType;
      }

      typeHolder(const std::string strValue, int type_ = noType){
        if(type_ == noType){
          if( parserNamespace::isAnInt(strValue.c_str()) )
            type = longType;
          else if( parserNamespace::isAFloat(strValue.c_str()) )
            type = doubleType;
          else if((strValue == "false") || (strValue == "true"))
            type = boolType;
        }
        else
          type = type_;

        switch(type){
        case intType   : value.int_    = toInt(strValue.c_str());    break;
        case boolType  : value.bool_   = toBool(strValue.c_str());   break;
        case charType  : value.char_   = toChar(strValue.c_str());   break;
        case longType  : value.long_   = toLong(strValue.c_str());   break;
        case shortType : value.short_  = toShort(strValue.c_str());  break;
        case floatType : value.float_  = toFloat(strValue.c_str());  break;
        case doubleType: value.double_ = toDouble(strValue.c_str()); break;
        default:
          std::cout << "Value not set\n";
          throw 1;
        }
      }

      bool isAFloat() const {
        switch(type){
        case intType   : return false; break;
        case boolType  : return false; break;
        case charType  : return false; break;
        case longType  : return false; break;
        case shortType : return false; break;
        case floatType : return true;  break;
        case doubleType: return true;  break;
        default:
          std::cout << "Value not set\n";
          throw 1;
        }
      }

      bool boolValue() const {
        switch(type){
        case intType   : return (bool) value.int_;    break;
        case boolType  : return (bool) value.bool_;   break;
        case charType  : return (bool) value.char_;   break;
        case longType  : return (bool) value.long_;   break;
        case shortType : return (bool) value.short_;  break;
        case floatType : return (bool) value.float_;  break;
        case doubleType: return (bool) value.double_; break;
        default:
          std::cout << "Value not set\n";
          throw 1;
        }
      }

      long longValue() const {
        switch(type){
        case intType   : return (long) value.int_;    break;
        case boolType  : return (long) value.bool_;   break;
        case charType  : return (long) value.char_;   break;
        case longType  : return (long) value.long_;   break;
        case shortType : return (long) value.short_;  break;
        case floatType : return (long) value.float_;  break;
        case doubleType: return (long) value.double_; break;
        default:
          std::cout << "Value not set\n";
          throw 1;
        }
      }

      double doubleValue() const {
        switch(type){
        case intType   : return (double) value.int_;    break;
        case boolType  : return (double) value.bool_;   break;
        case charType  : return (double) value.char_;   break;
        case longType  : return (double) value.long_;   break;
        case shortType : return (double) value.short_;  break;
        case floatType : return (double) value.float_;  break;
        case doubleType: return (double) value.double_; break;
        default:
          std::cout << "Value not set\n";
          throw 1;
        }
      }

      void setLongValue(const long &l){
        switch(type){
        case intType   : value.int_    = (int)    l; break;
        case boolType  : value.bool_   = (bool)   l; break;
        case charType  : value.char_   = (char)   l; break;
        case longType  : value.long_   = (long)   l; break;
        case shortType : value.short_  = (short)  l; break;
        case floatType : value.float_  = (float)  l; break;
        case doubleType: value.double_ = (double) l; break;
        default:
          std::cout << "Value not set\n";
          throw 1;
        }
      }

      void setDoubleValue(const double &d){
        switch(type){
        case intType   : value.int_    = (int)    d; break;
        case boolType  : value.bool_   = (bool)   d; break;
        case charType  : value.char_   = (char)   d; break;
        case longType  : value.long_   = (long)   d; break;
        case shortType : value.short_  = (short)  d; break;
        case floatType : value.float_  = (float)  d; break;
        case doubleType: value.double_ = (double) d; break;
        default:
          std::cout << "Value not set\n";
          throw 1;
        }
      }

      friend std::ostream& operator << (std::ostream &out, const typeHolder &th){
        switch(th.type){
        case intType   : out << th.value.int_;    break;
        case boolType  : out << th.value.bool_;   break;
        case charType  : out << th.value.char_;   break;
        case longType  : out << th.value.long_;   break;
        case shortType : out << th.value.short_;  break;
        case floatType : out << th.value.float_;  break;
        case doubleType: out << th.value.double_; break;
        default:
          std::cout << "Value not set\n";
          throw 1;
        }

        return out;
      }

      operator std::string () {
        std::stringstream ss;
        ss << *this;
        return ss.str();
      }
    };

    typeHolder applyOperator(std::string op, const typeHolder a){
      typeHolder ret;

      if(op == "!"){
        ret.type = boolType;
        if(a.isAFloat())
          ret.setDoubleValue( !a.doubleValue() );
        else
          ret.setLongValue( !a.longValue() );
      }
      else if(op == "+"){
        ret = a;
      }
      else if(op == "-"){
        ret = a;
        if(a.isAFloat())
          ret.setDoubleValue( -a.doubleValue() );
        else
          ret.setLongValue( -a.longValue() );
      }
      else if(op == "~"){
        ret.type = a.type;
        if(a.isAFloat()){
          std::cout << "Cannot apply [~] to [" << a << "].\n";
          throw 1;
        }
        else
          ret.setLongValue( ~a.longValue() );
      }

      return ret;
    }

    int typePrecedence(const typeHolder a, const typeHolder b){
      return ((a.type < b.type) ? b.type : a.type);
    }

    typeHolder applyOperator(const typeHolder a,
                             std::string op,
                             const typeHolder b){
      typeHolder ret;
      ret.type  = typePrecedence(a,b);
      ret.value = a.value;

      const bool aIsFloat = a.isAFloat();
      const bool bIsFloat = b.isAFloat();

      if(op == "+"){
        if(bIsFloat)
          ret.setDoubleValue(ret.doubleValue() + b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(ret.doubleValue() + b.longValue());
          else
            ret.setLongValue(ret.longValue() + b.longValue());
        }
      }
      else if(op == "-"){
        if(bIsFloat)
          ret.setDoubleValue(ret.doubleValue() - b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(ret.doubleValue() - b.longValue());
          else
            ret.setLongValue(ret.longValue() - b.longValue());
        }
      }
      else if(op == "*"){
        if(bIsFloat)
          ret.setDoubleValue(ret.doubleValue() * b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(ret.doubleValue() * b.longValue());
          else
            ret.setLongValue(ret.longValue() * b.longValue());
        }
      }
      else if(op == "/"){
        if(bIsFloat)
          ret.setDoubleValue(ret.doubleValue() / b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(ret.doubleValue() / b.longValue());
          else
            ret.setLongValue(ret.longValue() / b.longValue());
        }
      }

      else if(op == "+="){
        ret.type = a.type;

        if(bIsFloat)
          ret.setDoubleValue(ret.doubleValue() + b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(ret.doubleValue() + b.longValue());
          else
            ret.setLongValue(ret.longValue() + b.longValue());
        }
      }
      else if(op == "-="){
        ret.type = a.type;

        if(bIsFloat)
          ret.setDoubleValue(ret.doubleValue() - b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(ret.doubleValue() - b.longValue());
          else
            ret.setLongValue(ret.longValue() - b.longValue());
        }
      }
      else if(op == "*="){
        ret.type = a.type;

        if(bIsFloat)
          ret.setDoubleValue(ret.doubleValue() * b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(ret.doubleValue() * b.longValue());
          else
            ret.setLongValue(ret.longValue() * b.longValue());
        }
      }
      else if(op == "/="){
        ret.type = a.type;

        if(bIsFloat)
          ret.setDoubleValue(ret.doubleValue() / b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(ret.doubleValue() / b.longValue());
          else
            ret.setLongValue(ret.longValue() / b.longValue());
        }
      }

      else if(op == "<"){
        ret.type = boolType;

        if(bIsFloat){
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() < b.doubleValue());
          else
            ret.setLongValue(a.longValue() < b.doubleValue());
        }
        else{
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() < b.longValue());
          else
            ret.setLongValue(a.longValue() < b.longValue());
        }
      }
      else if(op == "<="){
        ret.type = boolType;

        if(bIsFloat){
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() <= b.doubleValue());
          else
            ret.setLongValue(a.longValue() <= b.doubleValue());
        }
        else{
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() <= b.longValue());
          else
            ret.setLongValue(a.longValue() <= b.longValue());
        }
      }
      else if(op == "=="){
        ret.type = boolType;

        if(bIsFloat){
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() == b.doubleValue());
          else
            ret.setLongValue(a.longValue() == b.doubleValue());
        }
        else{
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() == b.longValue());
          else
            ret.setLongValue(a.longValue() == b.longValue());
        }
      }
      else if(op == ">="){
        ret.type = boolType;

        if(bIsFloat){
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() >= b.doubleValue());
          else
            ret.setLongValue(a.longValue() >= b.doubleValue());
        }
        else{
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() >= b.longValue());
          else
            ret.setLongValue(a.longValue() >= b.longValue());
        }
      }
      else if(op == ">"){
        ret.type = boolType;

        if(bIsFloat){
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() > b.doubleValue());
          else
            ret.setLongValue(a.longValue() > b.doubleValue());
        }
        else{
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() > b.longValue());
          else
            ret.setLongValue(a.longValue() > b.longValue());
        }
      }
      else if(op == "!="){
        ret.type = boolType;

        if(bIsFloat){
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() != b.doubleValue());
          else
            ret.setLongValue(a.longValue() != b.doubleValue());
        }
        else{
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() != b.longValue());
          else
            ret.setLongValue(a.longValue() != b.longValue());
        }
      }

      else if(op == "="){
        ret.type = a.type;

        if(bIsFloat)
          ret.setDoubleValue( b.doubleValue() );
        else
          ret.setLongValue( b.longValue() );
      }

      else if(op == "<<"){
        if(bIsFloat){
          std::cout << "Cannot apply [A << B] where B is [float].\n";
          throw 1;
        }
        else if(aIsFloat){
          std::cout << "Cannot apply [A << B] where A is [float].\n";
          throw 1;
        }
        else
          ret.setLongValue(ret.longValue() << b.longValue());
      }
      else if(op == ">>"){
        if(bIsFloat){
          std::cout << "Cannot apply [A >> B] where B is [float].\n";
          throw 1;
        }
        else if(aIsFloat){
          std::cout << "Cannot apply [A >> B] where A is [float].\n";
          throw 1;
        }
        else
          ret.setLongValue(ret.longValue() >> b.longValue());
      }
      else if(op == "^"){
        if(bIsFloat){
          std::cout << "Cannot apply [A ^ B] where B is [float].\n";
          throw 1;
        }
        else if(aIsFloat){
          std::cout << "Cannot apply [A ^ B] where A is [float].\n";
          throw 1;
        }
        else
          ret.setLongValue(ret.longValue() ^ b.longValue());
      }
      else if(op == "|"){
        if(bIsFloat){
          std::cout << "Cannot apply [A | B] where B is [float].\n";
          throw 1;
        }
        else if(aIsFloat){
          std::cout << "Cannot apply [A | B] where A is [float].\n";
          throw 1;
        }
        else
          ret.setLongValue(ret.longValue() | b.longValue());
      }
      else if(op == "&"){
        if(bIsFloat){
          std::cout << "Cannot apply [A & B] where B is [float].\n";
          throw 1;
        }
        else if(aIsFloat){
          std::cout << "Cannot apply [A & B] where A is [float].\n";
          throw 1;
        }
        else
          ret.setLongValue(ret.longValue() & b.longValue());
      }
      else if(op == "%"){
        if(bIsFloat){
          std::cout << "Cannot apply [A % B] where B is [float].\n";
          throw 1;
        }
        else if(aIsFloat){
          std::cout << "Cannot apply [A % B] where A is [float].\n";
          throw 1;
        }
        else
          ret.setLongValue(ret.longValue() % b.longValue());
      }

      else if(op == "&&")
        ret.setLongValue( a.boolValue() && b.boolValue() );
      else if(op == "||")
        ret.setLongValue( a.boolValue() || b.boolValue() );

      else if(op == "%="){
        if(bIsFloat){
          std::cout << "Cannot apply [A % B] where B is [float].\n";
          throw 1;
        }
        else if(aIsFloat){
          std::cout << "Cannot apply [A % B] where A is [float].\n";
          throw 1;
        }
        else
          ret.setLongValue(ret.longValue() % b.longValue());
      }
      else if(op == "&="){
        if(bIsFloat){
          std::cout << "Cannot apply [A % B] where B is [float].\n";
          throw 1;
        }
        else if(aIsFloat){
          std::cout << "Cannot apply [A % B] where A is [float].\n";
          throw 1;
        }
        else
          ret.setLongValue(ret.longValue() % b.longValue());
      }
      else if(op == "^="){
        if(bIsFloat){
          std::cout << "Cannot apply [A % B] where B is [float].\n";
          throw 1;
        }
        else if(aIsFloat){
          std::cout << "Cannot apply [A % B] where A is [float].\n";
          throw 1;
        }
        else
          ret.setLongValue(ret.longValue() % b.longValue());
      }
      else if(op == "|="){
        if(bIsFloat){
          std::cout << "Cannot apply [A % B] where B is [float].\n";
          throw 1;
        }
        else if(aIsFloat){
          std::cout << "Cannot apply [A % B] where A is [float].\n";
          throw 1;
        }
        else
          ret.setLongValue(ret.longValue() % b.longValue());
      }

      return ret;
    }

    typeHolder applyOperator(const typeHolder a,
                             std::string op,
                             const typeHolder b,
                             const typeHolder c){
      bool pickC;

      if(a.isAFloat())
        pickC = (a.doubleValue() == 0);
      else
        pickC = (a.longValue() == 0);

      if(pickC)
        return c;
      else
        return b;
    }
    //==============================================


    //---[ Node ]-----------------------------------
    template <class TM>
    class node {
    public:
      node *left, *right, *up;
      std::vector< node<TM>* > down;
      TM value;

      inline node() :
        left(NULL),
        right(NULL),
        up(NULL) {}

      inline node(const TM &t) :
        left(NULL),
        right(NULL),
        up(NULL),

        value(t) {}

      inline node(const node<TM> &n) :
        left(n.left),
        right(n.right),
        up(n.up),
        down(n.down),

        value(n.value) {}

      inline node& operator = (const node<TM> &n){
        left  = n.left;
        right = n.right;
        up    = n.up;
        down  = n.down;

        value = n.value;
      }

      inline node<TM>* pop(){
        if(left != NULL)
          left->right = right;

        if(right != NULL)
          right->left = left;

        return this;
      }

      inline node* push(node<TM> *n){
        node *rr = right;

        right = n;

        right->left  = this;
        right->right = rr;

        if(rr)
          rr->left = right;

        return right;
      }

      inline node* push(const TM &t){
        return push(new node(t));
      }

      inline node* pushDown(node<TM> *n){
        n->up = this;

        down.push_back(n);

        return n;
      }

      inline node* pushDown(const TM &t){
        return pushDown(new node(t));
      }

      inline void print(const std::string &tab = ""){
        node *nodePos = this;

        while(nodePos){
          std::cout << tab << nodePos->value << '\n';

          const int downCount = (nodePos->down).size();

          if(downCount)
            printf("--------------------------------------------\n");

          for(int i = 0; i < downCount; ++i){
            (nodePos->down[i])->print(tab + "  ");
            printf("--------------------------------------------\n");
          }

          nodePos = nodePos->right;
        }
      }

      inline void printPtr(const std::string &tab = ""){
        node *nodePos = this;

        while(nodePos){
          std::cout << tab << *(nodePos->value) << '\n';

          const int downCount = (nodePos->down).size();

          if(downCount)
            printf("--------------------------------------------\n");

          for(int i = 0; i < downCount; ++i){
            (nodePos->down[i])->printPtr(tab + "  ");
            printf("--------------------------------------------\n");
          }

          nodePos = nodePos->right;
        }
      }
    };

    template <class TM>
    inline void popAndGoRight(node<TM> *&n){
      node<TM> *left  = n->left;
      node<TM> *right = n->right;

      if(left != NULL)
        left->right = right;

      if(right != NULL)
        right->left = left;

      delete n;

      n = right;
    }

    template <class TM>
    inline void popAndGoLeft(node<TM> *&n){
      node<TM> *left  = n->left;
      node<TM> *right = n->right;

      if(left != NULL)
        left->right = right;

      if(right != NULL)
        right->left = left;

      delete n;

      n = left;
    }

    template <class TM>
    node<TM>* firstNode(node<TM> *n){
      if((n == NULL) ||
         (n->left == NULL))
        return n;

      node<TM> *end = n;

      while(end->left)
        end = end->left;

      return end;
    }

    template <class TM>
    node<TM>* lastNode(node<TM> *n){
      if((n == NULL) ||
         (n->right == NULL))
        return n;

      node<TM> *end = n;

      while(end->right)
        end = end->right;

      return end;
    }

    class strNode {
    public:
      strNode *left, *right, *up;
      std::vector<strNode*> down;

      std::string value;
      int type, depth, sideDepth;

      inline strNode() :
        left(NULL),
        right(NULL),
        up(NULL),
        down(),

        value(""),

        type(0),
        depth(0),

        sideDepth(0) {}

      inline strNode(const std::string &value_) :
        left(NULL),
        right(NULL),
        up(NULL),
        down(),

        value(value_),

        type(0),
        depth(0),

        sideDepth(0) {}

      inline strNode(const strNode &n) :
        left(n.left),
        right(n.right),
        up(n.up),
        down(n.down),

        value(n.value),

        type(n.type),
        depth(n.depth),

        sideDepth(n.sideDepth) {}

      inline strNode& operator = (const strNode &n){
        left  = n.left;
        right = n.right;
        up    = n.up;
        down  = n.down;

        value = n.value;

        type  = n.type;
        depth = n.depth;

        sideDepth = n.sideDepth;

        return *this;
      }

      inline void swapWith(strNode *n){
        if(n == NULL)
          return;

        strNode *l1 = (left  == n) ? this : left;
        strNode *c1  = this;
        strNode *r1 = (right == n) ? this : right;

        strNode *l2 = (n->left  == this) ? n : n->left;
        strNode *c2 = right;
        strNode *r2 = (n->right == this) ? n : n->right;

        n->left  = l1;
        n->right = r1;

        left  = l2;
        right = r2;

        if(l1)
          l1->right = n;
        if(r1)
          r1->left  = n;

        if(l2)
          l2->right = this;
        if(r2)
          r2->left  = this;
      }

      inline void swapWithRight(){
        swapWith(right);
      }

      inline void swapWithLeft(){
        swapWith(left);
      }

      inline void moveLeftOf(strNode *n){
        if(n == NULL)
          return;

        if(left)
          left->right = right;
        if(right)
          right->left = left;

        left = n->left;

        if(n->left)
          left->right = this;

        right   = n;
        n->left = this;
      }

      inline void moveRightOf(strNode *n){
        if(n == NULL)
          return;

        if(left)
          left->right = right;
        if(right)
          right->left = left;

        right = n->right;

        if(n->right)
          right->left = this;

        left     = n;
        n->right = this;
      }

      inline strNode* clone() const {
        strNode *newNode = new strNode();

        newNode->value = value;
        newNode->type  = type;

        if(right){
          newNode->right = right->clone();
          newNode->right->left = newNode;
        }

        newNode->up = up;

        newNode->depth = depth;

        const int downCount = down.size();

        for(int i = 0; i < downCount; ++i)
          newNode->down.push_back( down[i]->clone() );

        return newNode;
      }

      inline operator std::string () const {
        return value;
      }

      strNode* pop();

      strNode* push(strNode *node);
      strNode* push(const std::string &str);

      strNode* pushDown(strNode *node);
      strNode* pushDown(const std::string &str);

      bool hasType(const int type_){
        if(type & type_)
          return true;

        const int downCount = down.size();

        for(int i = 0; i < downCount; ++i)
          if(down[i]->hasType(type_))
            return true;

        return false;
      }

      node<strNode*> getStrNodesWith(const std::string &name_,
                                     const int type_ = everythingType){
        node<strNode*> nRootNode;
        node<strNode*> *nNodePos = nRootNode.push(new strNode());

        strNode *nodePos = this;

        while(nodePos){
          if((nodePos->type & everythingType) &&
             (nodePos->value == name_)){

            nNodePos->value = nodePos;
            nNodePos = nNodePos->push(new strNode());
          }

          const int downCount = nodePos->down.size();

          for(int i = 0; i < downCount; ++i){
            node<strNode*> downRootNode = down[i]->getStrNodesWith(name_, type_);

            if(downRootNode.value != NULL){
              node<strNode*> *lastDownNode = (node<strNode*>*) downRootNode.value;

              node<strNode*> *nnpRight = nNodePos;
              nNodePos = nNodePos->left;

              nNodePos->right       = downRootNode.right;
              nNodePos->right->left = nNodePos;

              nNodePos = lastDownNode;
            }
          }

          nodePos = nodePos->right;
        }

        nNodePos = nNodePos->left;

        if(nNodePos == &nRootNode)
          nRootNode.value = NULL;
        else
          nRootNode.value = (strNode*) nNodePos;

        delete nNodePos->right;
        nNodePos->right = NULL;

        return nRootNode;
      }

      inline bool freeLeft(){
        if((left != NULL) && (left != this)){
          strNode *l = left;

          left = l->left;

          if(left != NULL)
            left->right = this;

          delete l;

          return true;
        }

        return false;
      }

      inline bool freeRight(){
        if((right != NULL) && (right != this)){
          strNode *r = right;

          right = r->right;

          if(right != NULL)
            right->left = this;

          delete r;

          return true;
        }

        return false;
      }

      void print(const std::string &tab = "");

      inline friend std::ostream& operator << (std::ostream &out, const strNode &n){
        out << n.value;
        return out;
      }
    };

    inline strNode* strNode::pop(){
      if(left != NULL)
        left->right = right;

      if(right != NULL)
        right->left = left;

      return this;
    }

    inline strNode* strNode::push(strNode *node){
      strNode *rr = right;

      right = node;

      right->left  = this;
      right->right = rr;
      right->up    = up;

      if(rr)
        rr->left = right;

      return right;
    }

    inline strNode* strNode::push(const std::string &value_){
      strNode *newNode = new strNode(value_);

      newNode->type      = type;
      newNode->depth     = depth;
      newNode->sideDepth = sideDepth;

      return push(newNode);
    };

    inline strNode* strNode::pushDown(strNode *node){
      node->up        = this;
      node->sideDepth = down.size();

      down.push_back(node);

      return node;
    };

    inline strNode* strNode::pushDown(const std::string &value_){
      strNode *newNode = new strNode(value_);

      newNode->type  = type;
      newNode->depth = depth + 1;

      return pushDown(newNode);
    };

    strNode* firstNode(strNode *node){
      if((node == NULL) ||
         (node->left == NULL))
        return node;

      strNode *end = node;

      while(end->left)
        end = end->left;

      return end;
    }

    strNode* lastNode(strNode *node){
      if((node == NULL) ||
         (node->right == NULL))
        return node;

      strNode *end = node;

      while(end->right)
        end = end->right;

      return end;
    }

    // autoMode: Handles newlines and tabs
    inline std::string prettyString(strNode *nodeRoot,
                                    const std::string &tab_ = "",
                                    const bool autoMode = true){
      strNode *nodePos = nodeRoot;

      std::string tab = tab_;
      std::string ret = "";

      while(nodePos){
        if(nodePos->type & operatorType){

          if(nodePos->type & binaryOperatorType){

            // char *blah
            if(nodePos->type & qualifierType){

              // [char ][*][blah]
              // or
              // [int ][a][ = ][0][, ][*][b][ = ][1][;]
              //                       ^
              if(nodePos->left &&
                 ((nodePos->left->type & descriptorType) ||
                  (nodePos->left->value == ","))){
                ret += *nodePos;

                // [const ][*][ const]
                if(nodePos->right &&
                   (nodePos->right->type & descriptorType) &&
                   !(nodePos->right->value == "*"))
                  ret += ' ';
              }
              else{
                ret += " ";
                ret += *nodePos;
                ret += " ";
              }
            }
            // [+] and [-]
            else if(nodePos->type & unitaryOperatorType){
              // (-blah ... )
              if(nodePos->left &&
                 !(nodePos->left->type & (presetValue | unknownVariable)) )
                ret += *nodePos;
              // a - b
              else{
                ret += " ";
                ret += *nodePos;
                ret += " ";
              }
            }
            else if(nodePos->value == ","){
              ret += ", ";
            }
            else if((nodePos->value == ".") || (nodePos->value == "::")){
              // [-] This check fails for current node loader
#if 0
              if(((nodePos->left == NULL) ||
                  !(nodePos->left->type & unknownVariable)) ||
                 ((nodePos->right == NULL) ||
                  !(nodePos->right->type & unknownVariable))){
                if(nodePos->left){
                  std::cout << "1. Error on:\n";
                  nodePos->left->print("  ");
                }
                else{
                  std::cout << "2. Error on:\n";
                  nodePos->print("  ");
                }

                throw 1;
              }
#endif

              ret += *nodePos;
            }
            else{
              ret += " ";
              ret += *nodePos;
              ret += " ";
            }

          }
          // [++] and [--]
          else if(nodePos->type & unitaryOperatorType){
            ret += *nodePos;
          }
          else if(nodePos->type & ternaryOperatorType){
            ret += " ? ";

            nodePos = nodePos->right;

            if((nodePos->right) == NULL){
              std::cout << "3. Error on:\n";
              nodePos->left->print("  ");
              throw 1;
            }

            if((nodePos->down).size())
              ret += prettyString(nodePos, "", autoMode);
            else
              ret += *nodePos;

            ret += " : ";

            if(((nodePos->right) == NULL)       ||
               ((nodePos->right->right) == NULL)){
              std::cout << "4. Error on:\n";
              nodePos->left->left->print("  ");
              throw 1;
            }

            nodePos = nodePos->right->right;

            if((nodePos->down).size())
              ret += prettyString(nodePos, "", autoMode);
            else
              ret += *nodePos;
          }
        }
        else if(nodePos->type & brace){
          if(nodePos->type & startSection){
            // a[] = {};
            if(nodePos->up->type & binaryOperatorType){
              ret += "{ ";
            }
            else{
              // Case: function(...) const {
              if( (((nodePos->sideDepth) != 0) &&
                   ((nodePos->up->down[nodePos->sideDepth - 1]->type & parentheses) ||
                    (nodePos->up->down[nodePos->sideDepth - 1]->value == "const")) )

                  || (nodePos->up->type & (occaKeywordType | flowControlType)))
                ret += " {\n" + tab + "  ";
              else
                ret += tab + "{\n";
            }

            tab += "  ";
          }
          else{
            tab = tab.substr(0, tab.size() - 2);

            // a[] = {};
            if(nodePos->up &&
               (nodePos->up->type & binaryOperatorType))
              ret += " }";
            else{
              ret += '}';

              //   }
              // }
              if((nodePos->up == NULL) ||
                 ((nodePos->up->right) &&
                  (nodePos->up->right->type == endBrace)))
                ret += "\n" + tab.substr(0, tab.size() - 2);
              else
                ret += "\n" + tab;
            }
          }
        }
        else if(nodePos->type == endParentheses){
          ret += ")";

          // if(...) statement
          if(autoMode)
            if((nodePos->up->type & flowControlType) &&
               (((nodePos->sideDepth) >= (nodePos->up->down.size() - 1)) ||
                !(nodePos->up->down[nodePos->sideDepth + 1]->type & brace))){

              ret += "\n" + tab + "  ";
            }
        }
        else if(nodePos->type & endStatement){
          ret += *nodePos;

          // for(){
          //   ...;
          // }
          if((nodePos->right == NULL) ||
             ((nodePos->right) &&
              (nodePos->right->type & brace))){

            ret += "\n" + tab.substr(0, tab.size() - 2);
          }
          //   blah;
          // }
          else if(!(nodePos->up)                    ||
                  !(nodePos->up->type & flowControlType) ||
                  !(nodePos->up->value == "for")){

            ret += "\n" + tab;
          }
          // Don't add newlines to for(A;B;C)
          else
            ret += " ";
        }
        else if(nodePos->type & descriptorType){
          ret += *nodePos;

          if(nodePos->right &&
             // [static ][const ][float ][variable]
             nodePos->right->type & (descriptorType |
                                     presetValue    |
                                     unknownVariable)){

            ret += " ";
          }
        }
        else if(nodePos->type & flowControlType){
          ret += *nodePos;

          if(autoMode)
            if(nodePos->down.size() == 0)
              ret += '\n' + tab + "  ";
        }
        else if(nodePos->type & specialKeywordType){
          if(nodePos->value == "case")
            ret += "case";
          else if(nodePos->value == "default")
            ret += "default";
          else if(nodePos->value == "break")
            ret += "break";
          else if(nodePos->value == "continue")
            ret += "continue";
          else if(nodePos->value == "return"){
            ret += "return";

            if(nodePos->right || nodePos->down.size())
              ret += ' ';
          }
          else if(nodePos->value == "goto")
            ret += "goto ";
          else
            ret += *nodePos;
        }
        else
          ret += *nodePos;

        const int downCount = (nodePos->down).size();

        for(int i = 0; i < downCount; ++i){
          strNode *downNode = nodePos->down[i];

          ret += prettyString(downNode, tab, autoMode);
        }

        nodePos = nodePos->right;
      }

      return ret;
    }

    inline void strNode::print(const std::string &tab){
      strNode *nodePos = this;

      while(nodePos){
        std::cout << tab << "[" << *nodePos << "] (" << nodePos->type << ")\n";

        const int downCount = (nodePos->down).size();

        if(downCount)
          printf("--------------------------------------------\n");

        for(int i = 0; i < downCount; ++i){
          (nodePos->down[i])->print(tab + "  ");
          printf("--------------------------------------------\n");
        }

        nodePos = nodePos->right;
      }
    }

    inline void popAndGoRight(strNode *&node){
      strNode *left  = node->left;
      strNode *right = node->right;

      if(left != NULL)
        left->right = right;

      if(right != NULL)
        right->left = left;

      delete node;

      node = right;
    }

    inline void popAndGoLeft(strNode *&node){
      strNode *left  = node->left;
      strNode *right = node->right;

      if(left != NULL)
        left->right = right;

      if(right != NULL)
        right->left = left;

      delete node;

      node = left;
    }

    inline void free(strNode *node){
      const int downCount = (node->down).size();

      for(int i = 0; i < downCount; ++i)
        free( (node->down)[i] );

      while(node->freeRight())
        /* Do Nothing */;

      while(node->freeLeft())
        /* Do Nothing */;

      delete node;
    }
    //==============================================


    //---[ Macro Info ]-----------------------------
    class macroInfo {
    public:
      std::string name;
      bool isAFunction;

      int argc;
      std::vector<std::string> parts;
      std::vector<int> argBetweenParts;

      inline macroInfo(){};

      inline std::string applyArgs(const std::vector<std::string> &args){
        if(argc != args.size()){
          if(args.size() == 0)
            printf("Macro [%s] uses [%d] arguments (None were provided).\n",
                   name.c_str(), argc);
          else if(args.size() == 1)
            printf("Macro [%s] uses [%d] arguments ([1] was provided).\n",
                   name.c_str(), argc);
          else
            printf("Macro [%s] uses [%d] arguments ([%d] were provided).\n",
                   name.c_str(), argc, (int) args.size());

          throw 1;
        }

        const int subs = argBetweenParts.size();

        std::string ret = parts[0];

        for(int i = 0; i < subs; ++i){
          const int argPos = argBetweenParts[i];
          ret += args[argPos];
          ret += parts[i + 1];
        }

        return ret;
      }

      inline friend std::ostream& operator << (std::ostream &out, const macroInfo &info){
        const int argc = info.argBetweenParts.size();

        out << info.name << ": " << info.parts[0];

        for(int i = 0; i < argc; ++i){
          const int argPos = info.argBetweenParts[i];
          out << "ARG" << argPos << info.parts[i + 1];
        }

        return out;
      }
    };
    //==============================================


    //---[ Statement ]------------------------------
    class varInfo;
    class statement;

    int statementType(strNode *&nodeRoot);
    varInfo loadVarInfo(strNode *&nodePos);

    class statement {
    public:
      varOriginMap_t &varOriginMap;
      varUsedMap_t &varUsedMap;

      int depth;
      statement *up;

      int type;

      strNode *nodeStart, *nodeEnd;

      scopeVarMap_t scopeVarMap;

      int statementCount;
      statementNode *statementStart, *statementEnd;

      inline statement(parserBase &pb) :
        depth(-1),
        type(blockStatementType),

        up(NULL),

        varOriginMap(pb.varOriginMap),
        varUsedMap(pb.varUsedMap),

        nodeStart(NULL),
        nodeEnd(NULL),

        statementCount(0),
        statementStart(NULL),
        statementEnd(NULL) {}

      inline ~statement(){};

      inline statement(const int depth_,
                       const int type_,
                       statement *up_,
                       strNode *nodeStart_, strNode *nodeEnd_) :
        depth(depth_),
        type(type_),

        up(up_),

        varOriginMap(up_->varOriginMap),
        varUsedMap(up_->varUsedMap),

        nodeStart(nodeStart_),
        nodeEnd(nodeEnd_),

        statementCount(0),
        statementStart(NULL),
        statementEnd(NULL) {}

      inline std::string getTab() const {
        std::string ret = "";

        for(int i = 0; i < depth; ++i)
          ret += "  ";

        return ret;
      }

      varInfo* hasVariableInScope(const std::string &varName){
        scopeVarMapIterator it = scopeVarMap.find(varName);

        if(it != scopeVarMap.end())
          return it->second;

        if(up)
          return up->hasVariableInScope(varName);

        return NULL;
      }

      bool hasDescriptorVariable(const std::string descriptor);
      bool hasDescriptorVariableInScope(const std::string descriptor);

      void loadAllFromNode(strNode *nodeRoot);
      strNode* loadFromNode(strNode *nodeRoot);

      strNode* loadSimpleFromNode(const int st,
                                  strNode *nodeRoot,
                                  strNode *nodeRootEnd);

      strNode* loadForFromNode(const int st,
                               strNode *nodeRoot,
                               strNode *nodeRootEnd);

      strNode* loadWhileFromNode(const int st,
                                 strNode *nodeRoot,
                                 strNode *nodeRootEnd);

      strNode* loadIfFromNode(const int st,
                              strNode *nodeRoot,
                              strNode *nodeRootEnd);

      // [-] Missing
      strNode* loadSwitchFromNode(const int st,
                                  strNode *nodeRoot,
                                  strNode *nodeRootEnd);

      strNode* loadGotoFromNode(const int st,
                                strNode *nodeRoot,
                                strNode *nodeRootEnd);

      strNode* loadFunctionDefinitionFromNode(const int st,
                                              strNode *nodeRoot,
                                              strNode *nodeRootEnd);

      strNode* loadFunctionPrototypeFromNode(const int st,
                                             strNode *nodeRoot,
                                             strNode *nodeRootEnd);

      strNode* loadBlockFromNode(const int st,
                                 strNode *nodeRoot,
                                 strNode *nodeRootEnd);

      // [-] Missing
      strNode* loadStructFromNode(const int st,
                                  strNode *nodeRoot,
                                  strNode *nodeRootEnd);

      // [-] Missing
      strNode* loadBlankFromNode(const int st,
                                 strNode *nodeRoot,
                                 strNode *nodeRootEnd);

      varInfo* addVariable(const varInfo &info,
                           statement *origin = NULL);

      void addStatement(statement *newStatement){
        if(statementStart != NULL){
          ++statementCount;
          statementEnd = statementEnd->push(newStatement);
        }
        else{
          statementCount = 1;
          statementStart = new node<statement*>(newStatement);
          statementEnd   = statementStart;
        }
      }

      statement* clone() const {
        statement *newStatement = new statement(depth,
                                                type, up,
                                                NULL, NULL);

        newStatement->nodeStart = nodeStart->clone();
        newStatement->nodeEnd   = lastNode(newStatement->nodeStart);

        newStatement->scopeVarMap = scopeVarMap;

        newStatement->statementCount = statementCount;

        newStatement->statementStart = NULL;
        newStatement->statementEnd   = NULL;

        if(statementCount == 0)
          return newStatement;

        statementNode *nodePos = statementStart;

        for(int i = 0; i < statementCount; ++i){
          newStatement->addStatement( nodePos->value->clone() );
          nodePos = nodePos->right;
        }

        return newStatement;
      }

      void printVariablesInStatement(){
        scopeVarMapIterator it = scopeVarMap.begin();

        while(it != scopeVarMap.end()){
          std::cout << "  " << it->first << '\n';

          ++it;
        }
      }

      void printVariablesInScope(){
        if(up)
          up->printVariablesInScope();

        printVariablesInStatement();
      }

      operator std::string() const {
        const std::string tab = getTab();

        statementNode *statementPos = statementStart;

        // OCCA For's
        if(type == (occaStatementType | forStatementType)){
          std::string ret = tab + nodeStart->value + " {\n";

          while(statementPos){
            ret += (std::string) *(statementPos->value);
            statementPos = statementPos->right;
          }

          ret += tab + "}\n";

          return ret;
        }
        else if(type & (simpleStatementType | gotoStatementType)){
          return tab + prettyString(nodeStart, "", false);
        }

        else if(type & flowStatementType){
          std::string ret = tab + prettyString(nodeStart, "", false);

          if(statementCount > 1)
            ret += " {";

          ret += '\n';

          while(statementPos){
            ret += (std::string) *(statementPos->value);
            statementPos = statementPos->right;
          }

          if(statementCount > 1)
            ret += tab + "}\n";

          return ret;
        }

        else if(type & functionStatementType){
          if(type & functionDefinitionType){
            std::string ret = prettyString(nodeStart, "", false);

            ret += " {\n";

            while(statementPos){
              ret += (std::string) *(statementPos->value);
              statementPos = statementPos->right;
            }

            ret += tab + "}\n";

            return ret;
          }
          else if(type & functionPrototypeType)
            return tab + prettyString(nodeStart, "", false);
        }
        else if(type & blockStatementType){
          std::string ret = "";

          if(0 <= depth)
            ret += tab + "{\n";

          while(statementPos){
            ret += (std::string) *(statementPos->value);
            statementPos = statementPos->right;
          }

          if(0 <= depth)
            ret += tab + "}\n";

          return ret;
        }
        else if(type & structStatementType)
          return tab + prettyString(nodeStart, "", false);

        return tab + prettyString(nodeStart, "", false);
      }
    };

    inline std::ostream& operator << (std::ostream &out, const statement &s){
      out << (std::string) s;

      return out;
    }
    //==============================================


    //---[ Variable Info ]--------------------------
    class varInfo {
    public:
      std::string type, name;
      int typeInfo;

      int pointerCount;
      std::vector<std::string> descriptors;
      std::vector<std::string> stackPointerSizes;

      std::vector<std::string> extraInfo;

      inline varInfo() :
        type(""),
        name(""),
        typeInfo(0),

        pointerCount(0) {};

      inline varInfo(const varInfo &vi){
        type     = vi.type;
        name     = vi.name;
        typeInfo = vi.typeInfo;

        pointerCount = vi.pointerCount;

        descriptors       = vi.descriptors;
        stackPointerSizes = vi.stackPointerSizes;

        extraInfo = vi.extraInfo;
      }

      inline bool hasDescriptor(const std::string descriptor) const {
        const int descriptorCount = descriptors.size();

        for(int i = 0; i < descriptorCount; ++i){
          if(descriptors[i] == descriptor)
            return true;
        }

        return false;
      }

      inline strNode* makeStrNodeChain(const int depth     = 0,
                                       const int sideDepth = 0) const {
        strNode *nodeRoot = new strNode();
        strNode *nodePos = nodeRoot;

        nodeRoot->depth     = depth;
        nodeRoot->sideDepth = sideDepth;

        const int descriptorCount = descriptors.size();

        for(int i = 0; i < descriptorCount; ++i){
          nodePos       = nodePos->push(descriptors[i]);
          nodePos->type = qualifierType;
        }

        if(type.size()){
          nodePos       = nodePos->push(type);
          nodePos->type = specifierType;
        }

        if(typeInfo & pointerType){
          if(typeInfo & heapPointerType){
            for(int i = 0; i < pointerCount; ++i){
              nodePos       = nodePos->push("*");
              nodePos->type = keywordType["*"];
            }
          }

          if(typeInfo & constPointerType){
            nodePos       = nodePos->push("const");
            nodePos->type = keywordType["const"];
          }
        }

        if(typeInfo & referenceType){
          nodePos       = nodePos->push("&");
          nodePos->type = keywordType["&"];
        }

        nodePos       = nodePos->push(name);
        nodePos->type = unknownVariable;

        if(typeInfo & stackPointerType){
          const int heapCount = stackPointerSizes.size();

          for(int i = 0; i < heapCount; ++i){
            strNode *downNode = nodePos->pushDown("[");
            downNode->type    = keywordType["["];

            downNode       = downNode->push(stackPointerSizes[i]);
            downNode->type = unknownVariable; // [-] Quick fix

            downNode       = downNode->push("[");
            downNode->type = keywordType["]"];
          }
        }

        if(typeInfo & gotoType){
          nodePos       = nodePos->push(":");
          nodePos->type = keywordType[":"];
        }
        else{
          nodePos       = nodePos->push(";");
          nodePos->type = keywordType[";"];
        }

        popAndGoRight(nodeRoot);

        return nodeRoot;
      }
    };

    inline std::ostream& operator << (std::ostream &out, const varInfo &info){
      const int descriptorCount = info.descriptors.size();

      for(int i = 0; i < descriptorCount; ++i)
        out << info.descriptors[i] << ' ';

      if(info.type.size())
        out << info.type << ' ';

      if(info.typeInfo & pointerType){
        if(info.typeInfo & heapPointerType){
          for(int i = 0; i < info.pointerCount; ++i)
            out << '*';
        }

        if(info.typeInfo & constPointerType)
          out << " const ";
      }

      if(info.typeInfo & referenceType)
        out << '&';

      out << info.name;

      if(info.typeInfo & stackPointerType){
        const int heapCount = info.stackPointerSizes.size();

        for(int i = 0; i < heapCount; ++i)
          out << '[' << info.stackPointerSizes[i] << ']';
      }

      if(info.typeInfo & gotoType)
        out << ':';

      return out;
    }
    //==============================================


    //---[ Helper Functions ]-----------------------
    std::string obfuscate(const std::string s1){
      return "__occa__variable__" + s1 + "__";
    }

    std::string obfuscate(const std::string s1, const std::string s2){
      return "__occa__variable__" + s1 + "__" + s2 + "__";
    }

    bool stringsAreEqual(const char *cStart, const size_t chars,
                         const char *c2){
      for(size_t c = 0; c < chars; ++c){
        if(cStart[c] != c2[c])
          return false;

        if(c2[c] == '\0')
          return false;
      }

      return true;
    }

    bool charIsIn(const char c, const char *delimeters){
      while((*delimeters) != '\0')
        if(c == *(delimeters++))
          return true;

      return false;
    }

    bool charIsIn2(const char *c, const char *delimeters){
      const char c0 = c[0];
      const char c1 = c[1];

      while((*delimeters) != '\0'){
        if((c0 == delimeters[0]) && (c1 == delimeters[1]))
          return true;

        delimeters += 2;
      }

      return false;
    }

    bool isWhitespace(const char c){
      return charIsIn(c, whitespace);
    }

    void skipWhitespace(const char *&c){
      while(charIsIn(*c, whitespace) && (*c != '\0'))
        ++c;
    }

    void skipToWhitespace(const char *&c){
      while(!charIsIn(*c, whitespace) && (*c != '\0'))
        ++c;
    }

    bool isAString(const char *c){
      return ((*c == '\'') || (*c == '"'));
    }

    bool isAnInt(const char *c){
      const char *cEnd = c;
      skipToWhitespace(cEnd);

      while(c < cEnd){
        if(('0' > *c) || (*c > '9'))
          return false;

        ++c;
      }

      return true;
    }

    bool isAFloat(const char *c){
      if(('0' <= *c) && (*c <= '9'))
        return true;

      if(((c[0] == '+') || (c[0] == '-')) &&
         ((c + 1) != '\0') &&
         ((c[1] == '.') || (('0' <= c[1]) && (c[1] <= '9'))))
        return true;

      if((c[0] == '.')     &&
         ((c + 1) != '\0') &&
         ('0' <= c[1]) && (c[1] <= '9'))
        return true;

      return false;
    }

    bool isANumber(const char *c){
      return (isAnInt(c) || isAFloat(c));
    }

    void skipInt(const char *&c){
      while((c != '\0') &&
            ('0' <= *c) && (*c <= '9'))
        ++c;
    }

    void skipNumber(const char *&c){
      if((*c == '+') || (*c == '-'))
        ++c;

      skipInt(c);

      if(*c == '.')
        ++c;

      skipInt(c);

      if(*c == 'e')
        ++c;

      if((*c == '+') || (*c == '-'))
        ++c;

      skipInt(c);

      if(*c == 'f')
        ++c;
    }

    void skipString(const char *&c){
      if(!isAString(c))
        return;

      const char match = *(c++);

      while(*c != '\0'){
        if(*c == '\\')
          ++c;
        else if(*c == match){
          ++c;
          return;
        }

        ++c;
      }
    }

    char isAWordDelimeter(const char *c){
      if( charIsIn(c[0], wordDelimeter) ){
        if(charIsIn2(c, wordDelimeterExt))
          return 2;

        return 1;
      }

      return 0;
    }

    int skipWord(const char *&c){
      while(!charIsIn(*c, whitespace) && (*c != '\0')){
        const int delimeterChars = isAWordDelimeter(c);

        if(delimeterChars == 0)
          ++c;
        else
          return delimeterChars;
      }

      return 0;
    }

    const char* readLine(const char *c){
      const char *c0 = c;
      bool breakNextLine = true;

      while(*c != '\0'){
        skipString(c);

        if(*c == '\0')
          break;

        if(*c == '\n'){
          if(breakNextLine)
            break;

          breakNextLine = false;
        }
        else if((c[0] == '\\') && isWhitespace(c[1])){
          breakNextLine = true;
          ++c;
        }

        ++c;
      }

      return (c + 1);
    }

    std::string compressWhitespace(const std::string &str){
      const size_t chars = str.size();
      std::string ret = str;

      const char *c = str.c_str();
      size_t pos = 0;

      while(*c != '\0'){
        if(isWhitespace(*c)){
          ret[pos++] = ' ';

          skipWhitespace(c);
        }
        else
          ret[pos++] = *(c++);
      }

      ret.resize(pos);

      return ret;
    }

    std::string strip(const char *c, const size_t chars){
      if(chars == 0)
        return "";

      const char *cLeft  = c;
      const char *cRight = c + (chars - 1);

      while(charIsIn(*cLeft , whitespace) && (cLeft <= cRight)) ++cLeft;
      while(charIsIn(*cRight, whitespace) && (cRight > cLeft)) --cRight;

      if(cLeft > cRight)
        return "";

      std::string ret = "";

      const char *cMid = cLeft;

      while(cMid < cRight){
        if((cMid[0] == '\\') && isWhitespace(cMid[1])){
          ret += strip(cLeft, cMid - cLeft);
          ++cMid;

          cLeft = (cMid + 1);
        }

        ++cMid;

        if((cMid >= cRight) && ret.size())
          ret += strip(cLeft, (cMid - cLeft + 1));
      }

      if(ret.size() == 0)
        return compressWhitespace( std::string(cLeft, (cRight - cLeft + 1)) );

      return compressWhitespace(ret);
    }

    void strip(std::string &str){
      str = strip(str.c_str(), str.size());
    }

    char* cReadFile(const std::string &filename){
      int fileHandle = ::open(filename.c_str(), O_RDWR);

      if(fileHandle == 0){
        printf("File [ %s ] does not exist.\n", filename.c_str());
        throw 1;
      }

      struct stat fileInfo;
      const int status = fstat(fileHandle, &fileInfo);

      if(status != 0){
        printf( "File [ %s ] gave a bad fstat.\n" , filename.c_str());
        throw 1;
      }

      const uintptr_t chars = fileInfo.st_size;

      char *fileContents = new char[chars + 1];
      fileContents[chars] = '\0';

      ::read(fileHandle, fileContents, chars);

      ::close(fileHandle);

      return fileContents;
    }

    int stripComments(std::string &line){
      std::string line2  = line;
      line = "";

      const char *cLeft  = line2.c_str();
      const char *cRight = cLeft;

      int status = readingCode;

      while(cRight != '\0'){
        skipString(cRight);

        if((*cRight == '\0') || (*cRight == '\n'))
          break;

        if((cRight[0] == '/') && (cRight[1] == '/')){
          line += std::string(cLeft, cRight - cLeft);
          return readingCode;
        }
        else if((cRight[0] == '/') && (cRight[1] == '*')){
          line += std::string(cLeft, cRight - cLeft);
          status = insideCommentBlock;
          cLeft = cRight + 2;
        }
        else if((cRight[0] == '*') && (cRight[1] == '/')){
          if(status == insideCommentBlock)
            status = readingCode;
          else
            status = finishedCommentBlock;
          cLeft = cRight + 2;
        }

        ++cRight;
      }

      if(cLeft != cRight)
        line += std::string(cLeft, cRight - cLeft);

      return status;
    }
    //==============================================


    //---[ Statement Functions ]--------------------{
    inline bool statement::hasDescriptorVariable(const std::string descriptor){
      scopeVarMapIterator it = scopeVarMap.begin();

      while(it != scopeVarMap.end()){
        if((it->second)->hasDescriptor(descriptor))
          return true;

        ++it;
      }

      return false;
    }

    inline bool statement::hasDescriptorVariableInScope(const std::string descriptor){
      if(hasDescriptorVariable(descriptor))
        return true;

      if(up != NULL)
        return up->hasDescriptorVariable(descriptor);

      return false;
    }

    inline varInfo* statement::addVariable(const varInfo &info,
                                           statement *origin){
      scopeVarMapIterator it = scopeVarMap.find(info.name);
      if(it != scopeVarMap.end()       &&
         !info.hasDescriptor("extern") &&
         !((info.typeInfo & functionType) && ((it->second)->typeInfo & protoType))){

        std::cout << "Variable [" << info.name << "] already defined on:"
                  << *this;
        throw 1;
      }

      varInfo *&newInfo = scopeVarMap[info.name];

      newInfo = new varInfo(info);

      if(origin == NULL)
        varOriginMap[newInfo] = this;
      else{
        varOriginMap[newInfo]          = origin;
        origin->scopeVarMap[info.name] = newInfo;
      }

      return newInfo;
    }

    inline void statement::loadAllFromNode(strNode *nodeRoot){
      while(nodeRoot)
        nodeRoot = loadFromNode(nodeRoot);
    }

    inline strNode* statement::loadFromNode(strNode *nodeRoot){
      strNode *nodeRootEnd = nodeRoot;

      // Finds statement type and sets nodeRootEnd to the
      //    last strNode in that statement
      const int st = statementType(nodeRootEnd);

      statement *newStatement = new statement(depth + 1,
                                              st, this,
                                              nodeRoot, nodeRootEnd);

      if(st & simpleStatementType){
        nodeRootEnd = newStatement->loadSimpleFromNode(st,
                                                       nodeRoot,
                                                       nodeRootEnd);
      }

      else if(st & flowStatementType){
        if(st & forStatementType)
          nodeRootEnd = newStatement->loadForFromNode(st,
                                                      nodeRoot,
                                                      nodeRootEnd);

        else if(st & whileStatementType)
          nodeRootEnd = newStatement->loadWhileFromNode(st,
                                                        nodeRoot,
                                                        nodeRootEnd);

        else if(st & ifStatementType){
          delete newStatement;
          newStatement = NULL;

          nodeRootEnd = loadIfFromNode(st,
                                       nodeRoot,
                                       nodeRootEnd);
        }

        else if(st & switchStatementType)
          nodeRootEnd = newStatement->loadSwitchFromNode(st,
                                                         nodeRoot,
                                                         nodeRootEnd);

        else if(st & gotoStatementType)
          nodeRootEnd = newStatement->loadGotoFromNode(st,
                                                       nodeRoot,
                                                       nodeRootEnd);
      }
      else if(st & functionStatementType){
        if(st & functionDefinitionType)
          nodeRootEnd = newStatement->loadFunctionDefinitionFromNode(st,
                                                                     nodeRoot,
                                                                     nodeRootEnd);

        else if(st & functionPrototypeType)
          nodeRootEnd = newStatement->loadFunctionPrototypeFromNode(st,
                                                                    nodeRoot,
                                                                    nodeRootEnd);
      }
      else if(st & blockStatementType)
        nodeRootEnd = newStatement->loadBlockFromNode(st,
                                                      nodeRoot,
                                                      nodeRootEnd);

      else if(st & structStatementType)
        nodeRootEnd = newStatement->loadStructFromNode(st,
                                                       nodeRoot,
                                                       nodeRootEnd);

      else if(st & blankStatementType)
        nodeRootEnd = newStatement->loadBlankFromNode(st,
                                                      nodeRoot,
                                                      nodeRootEnd);

      if(newStatement)
        addStatement(newStatement);

      return nodeRootEnd;
    }

    inline strNode* statement::loadSimpleFromNode(const int st,
                                                  strNode *nodeRoot,
                                                  strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    inline strNode* statement::loadForFromNode(const int st,
                                               strNode *nodeRoot,
                                               strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      const int downCount = nodeRootEnd->down.size();

      if(downCount == 1){
        // for(;;)
        //   stuff;
        if(nodeRoot->down[0]->type == startParentheses)
          return loadFromNode(nextNode);

        // occaOuterFor {
        // }
        else{
          strNode *blockStart = nodeRootEnd->down[0];
          strNode *blockEnd   = lastNode(blockStart);

          nodeRootEnd->down.erase(nodeRootEnd->down.begin() + 0,
                                  nodeRootEnd->down.begin() + 1);

          // Load all down's before popping [{] and [}]'s
          const int downCount = blockStart->down.size();

          for(int i = 0; i < downCount; ++i)
            loadAllFromNode( blockStart->down[i] );

          popAndGoRight(blockStart);
          popAndGoLeft(blockEnd);

          loadAllFromNode(blockStart);
        }
      }
      else{
        strNode *blockStart = nodeRoot->down[1];
        strNode *blockEnd   = lastNode(blockStart);

        nodeRoot->down.erase(nodeRoot->down.begin() + 1,
                             nodeRoot->down.begin() + 2);

        // Load all down's before popping [{] and [}]'s
        const int downCount = blockStart->down.size();

        for(int i = 0; i < downCount; ++i)
          loadAllFromNode( blockStart->down[i] );

        popAndGoRight(blockStart);
        popAndGoLeft(blockEnd);

        loadAllFromNode(blockStart);
      }

      return nextNode;
    }

    inline strNode* statement::loadWhileFromNode(const int st,
                                                 strNode *nodeRoot,
                                                 strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      const int downCount = nodeRootEnd->down.size();

      if(downCount == 1)
        return loadFromNode(nextNode);

      else{
        strNode *blockStart = nodeRoot->down[1];
        strNode *blockEnd   = lastNode(blockStart);

        nodeRoot->down.erase(nodeRoot->down.begin() + 1,
                             nodeRoot->down.begin() + 2);

        // Load all down's before popping [{] and [}]'s
        const int downCount = blockStart->down.size();

        for(int i = 0; i < downCount; ++i)
          loadAllFromNode( blockStart->down[i] );

        popAndGoRight(blockStart);
        popAndGoLeft(blockEnd);

        loadAllFromNode(blockStart);
      }

      return nextNode;
    }

    inline strNode* statement::loadIfFromNode(const int st_,
                                              strNode *nodeRoot,
                                              strNode *nodeRootEnd){
      int st = st_;
      strNode *nextNode;

      do {
        nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

        if(nodeRoot)
          nodeRoot->left = NULL;
        if(nodeRootEnd)
          nodeRootEnd->right = NULL;

        statement *newStatement = new statement(depth + 1,
                                                st, this,
                                                nodeRoot, nodeRootEnd);

        addStatement(newStatement);

        const int downCount = nodeRootEnd->down.size();

        if((downCount == 1) ||
           ((downCount == 0) && (st == elseStatementType))){
          // for(;;)    or    else
          //   statement;       statement;

          nextNode = newStatement->loadFromNode(nextNode);

          if(st == elseStatementType)
            break;
        }
        else{
          int blockPos = (st != elseStatementType) ? 1 : 0;

          strNode *blockStart = nodeRoot->down[blockPos];
          strNode *blockEnd   = lastNode(blockStart);

          // Load all down's before popping [{] and [}]'s
          const int blockDownCount = blockStart->down.size();

          for(int i = 0; i < blockDownCount; ++i)
            loadAllFromNode( blockStart->down[i] );

          for(int i = (blockPos + 1); i < downCount; ++i)
            loadAllFromNode(nodeRoot->down[i]);

          nodeRoot->down.clear();

          popAndGoRight(blockStart);
          popAndGoLeft(blockEnd);

          newStatement->loadAllFromNode(blockStart);

          break;
        }

        if(nextNode == NULL)
          break;

        nodeRoot = nodeRootEnd = nextNode;

        st = statementType(nodeRootEnd);

      } while((st == elseIfStatementType) ||
              (st == elseStatementType));

      return nextNode;
    }

    // [-] Missing
    inline strNode* statement::loadSwitchFromNode(const int st,
                                                  strNode *nodeRoot,
                                                  strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    inline strNode* statement::loadGotoFromNode(const int st,
                                                strNode *nodeRoot,
                                                strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    inline strNode* statement::loadFunctionDefinitionFromNode(const int st,
                                                              strNode *nodeRoot,
                                                              strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      strNode *blockStart = nodeRootEnd->down[1];
      strNode *blockEnd   = lastNode(blockStart);

      nodeRootEnd->down.erase(nodeRootEnd->down.begin() + 1,
                              nodeRootEnd->down.begin() + 2);

      // Load all down's before popping [{] and [}]'s
      const int downCount = blockStart->down.size();

      for(int i = 0; i < downCount; ++i)
        loadAllFromNode( blockStart->down[i] );

      if(blockStart->right == blockEnd){
        popAndGoRight(blockStart);
        popAndGoLeft(blockEnd);

        return nextNode;
      }

      popAndGoRight(blockStart);
      popAndGoLeft(blockEnd);

      loadAllFromNode(blockStart);

      return nextNode;
    }

    inline strNode* statement::loadFunctionPrototypeFromNode(const int st,
                                                             strNode *nodeRoot,
                                                             strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    inline strNode* statement::loadBlockFromNode(const int st,
                                                 strNode *nodeRoot,
                                                 strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      // Blocks don't have stuff, they just provide a new scope
      //   Hence, nodeStart = nodeEnd = NULL
      nodeStart = nodeEnd = NULL;

      // Load all down's before popping [{] and [}]'s
      const int downCount = nodeRoot->down.size();

      for(int i = 0; i < downCount; ++i)
        loadAllFromNode( nodeRoot->down[i] );

      if(nodeRoot->right == nodeRootEnd){
        popAndGoRight(nodeRoot);
        popAndGoLeft(nodeRootEnd);

        return nextNode;
      }

      popAndGoRight(nodeRoot);
      popAndGoLeft(nodeRootEnd);

      loadAllFromNode(nodeRoot);

      return nextNode;
    }

    // [-] Missing
    inline strNode* statement::loadStructFromNode(const int st,
                                                  strNode *nodeRoot,
                                                  strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    // [-] Missing
    inline strNode* statement::loadBlankFromNode(const int st,
                                                 strNode *nodeRoot,
                                                 strNode *nodeRootEnd){
      strNode *nextNode = nodeRootEnd ? nodeRootEnd->right : NULL;

      if(nodeRoot)
        nodeRoot->left = NULL;
      if(nodeRootEnd)
        nodeRootEnd->right = NULL;

      return nextNode;
    }

    inline void parserBase::applyToAllStatements(statement &s,
                                                 applyToAllStatements_t func){
      (this->*func)(s);

      statementNode *statementPos = s.statementStart;

      while(statementPos){
        applyToAllStatements(*(statementPos->value), func);
        statementPos = statementPos->right;
      }
    }

    inline void parserBase::applyToStatementsDefiningVar(applyToStatementsDefiningVar_t func){
      varOriginMapIterator it = varOriginMap.begin();

      while(it != varOriginMap.end()){
        (this->*func)(*(it->first), *(it->second));

        ++it;
      }
    }

    inline void parserBase::applyToStatementsUsingVar(varInfo &info,
                                                      applyToStatementsUsingVar_t func){
      varUsedMapIterator it = varUsedMap.find(&info);

      if(it != varUsedMap.end()){
        statementNode *sn = it->second.right;

        while(sn){
          (this->*func)(info, *(sn->value));

          sn = sn->right;
        }
      }
    }

    inline bool parserBase::statementIsAKernel(statement &s){
      if(s.type & functionStatementType){
        strNode *nodePos = s.nodeStart;

        while(nodePos){
          if(nodePos->value == "occaKernel")
            return true;

          nodePos = nodePos->right;
        }
      }

      return false;
    }

    inline statement* parserBase::getStatementKernel(statement &s){
      statement *sUp = &s;

      while(sUp){
        if(statementIsAKernel(*sUp))
          return sUp;

        sUp = sUp->up;
      }

      return sUp;
    }

    inline bool parserBase::statementKernelUsesNativeOCCA(statement &s){
      statement &sKernel = *(getStatementKernel(s));

      std::string check = obfuscate("native", "occa");
      varInfo *info = sKernel.hasVariableInScope(check);

      if(info != NULL)
        return true;
      else
        return false;
    }

    inline bool parserBase::statementKernelUsesNativeOKL(statement &s){
      statement &sKernel = *(getStatementKernel(s));

      std::string check = obfuscate("native", "okl");
      varInfo *info = sKernel.hasVariableInScope(check);

      if(info != NULL)
        return true;
      else
        return false;
    }

    inline bool parserBase::statementKernelUsesNativeLanguage(statement &s){
      if(statementKernelUsesNativeOKL(s))
        return true;

      if(statementKernelUsesNativeOCCA(s))
        return true;

      return false;
    }

    inline void parserBase::addOccaForCounter(statement &s,
                                              const std::string &ioLoop,
                                              const std::string &loopNest){
      varInfo ioDimVar;
      ioDimVar.name = obfuscate(ioLoop);
      ioDimVar.extraInfo.push_back(loopNest);

      varInfo *ioDimVar2 = s.hasVariableInScope(ioDimVar.name);

      if(ioDimVar2 == NULL){
        statement &sKernel = *(getStatementKernel(s));
        sKernel.addVariable(ioDimVar);
      }
      else{
        const int extras = ioDimVar2->extraInfo.size();
        int i;

        for(i = 0; i < extras; ++i){
          if(ioDimVar2->extraInfo[i] == loopNest)
            break;
        }

        if(i == extras)
          ioDimVar2->extraInfo.push_back(loopNest);
      }
    }

    inline bool parserBase::nodeHasUnknownVariable(strNode *n){
      while(n){
        if(n->type & unknownVariable)
          return true;

        const int downCount = n->down.size();

        for(int i = 0; i < downCount; ++i)
          if(nodeHasUnknownVariable(n->down[i]))
            return true;

        n = n->right;
      }

      return false;
    }

    inline void parserBase::setupOccaFors(statement &s){
      if( !(s.type & forStatementType) )
        return;

      if(statementKernelUsesNativeOCCA(s))
        return;

      statement &sKernel = *(getStatementKernel(s));

      strNode *nodePos = s.nodeStart;
      strNode *lastNode;

      while(nodePos->down.size() == 0)
        nodePos = nodePos->right;

      lastNode = nodePos;
      nodePos  = nodePos->down[0]->right;

      std::vector<strNode*> oldDown(lastNode->down.begin() + 1,
                                    lastNode->down.end());

      // Last segment doesn't have a [;]
      int segmentCount = 1;
      strNode *commaNodes[4];

      // First statement
      commaNodes[0] = nodePos;

      while(nodePos){
        if(nodePos->value == ";"){
          if(segmentCount == 4){
            std::cout << "More than 4 statements for:\n  " << prettyString(s.nodeStart) << '\n';
            throw 1;
          }

          commaNodes[segmentCount++] = nodePos;
        }

        nodePos = nodePos->right;
      }

      if(segmentCount < 4)
        return;

      nodePos = commaNodes[3]->right;

      // If it has a fourth argument, make sure it's the correct one
      if( ((nodePos->value.find("inner") == std::string::npos) ||
           ((nodePos->value != "inner0") &&
            (nodePos->value != "inner1") &&
            (nodePos->value != "inner2")))                     &&
          ((nodePos->value.find("outer") == std::string::npos) ||
           ((nodePos->value != "outer0") &&
            (nodePos->value != "outer1") &&
            (nodePos->value != "outer2"))) ){

        std::cout << "Wrong 4th statement for:\n  " << prettyString(s.nodeStart) << '\n';
        throw 1;
      }

      // [-----][#]
      std::string ioLoop   = nodePos->value.substr(0,5);
      std::string loopNest = nodePos->value.substr(5,1);

      ioLoop[0] += ('A' - 'a');

      addOccaForCounter(s, ioLoop, loopNest);

      varInfo idVar, dimVar, loopVar;
      std::string idName, dimName, dimSubName, loopName;

      nodePos = commaNodes[0];
      idVar = loadVarInfo(nodePos);

      idName = idVar.name;
      idVar.extraInfo.push_back("occa" + ioLoop + "Id" + loopNest);

      if(nodePos->value != "="){
        std::cout << "The first statement of:\n"
                  << "  " << prettyString(s.nodeStart, "  ", false) << '\n'
                  << "should look like:\n"
                  << "  for(int var = off; ...; ...; ...)\n";
        throw 1;
      }

      nodePos = nodePos->right;

      strNode *idOffStart = nodePos->clone();
      strNode *idOffEnd   = idOffStart;

      while(nodePos != commaNodes[1]){
        if( nodePos->hasType(unknownVariable) ){
          std::cout << "The first statement of:\n"
                    << "  " << prettyString(s.nodeStart, "  ", false) << '\n'
                    << "should look like:\n"
                    << "  for(int var = off; ...; ...; ...)\n"
                    << "where [off] needs to be known at compile time\n";
          throw 1;
        }

        nodePos  = nodePos->right;
        idOffEnd = idOffEnd->right;
      }

      // Go behind [;]
      idOffEnd = idOffEnd->left;
      // Don't kill me
      idOffEnd->left = NULL;
      free(idOffEnd);

      idVar.extraInfo.push_back( prettyString(idOffStart) );

      // [-] ?
      if(!sKernel.hasVariableInScope(idVar.name))
        sKernel.addVariable(idVar);

      s.addVariable(idVar);

      nodePos = nodePos->right;

      if(nodePos->value != idName){
        std::cout << "The second statement of:\n"
                  << "  " << prettyString(s.nodeStart, "  ", false) << '\n'
                  << "should look like:\n"
                  << "  for(int [var] = off; [var] < dim; ...; ...)\n"
                  << "           ^____________^ are the same\n";
        throw 1;
      }

      nodePos = nodePos->right;

      if(nodePos->value != "<"){
        std::cout << "The second statement of:\n"
                  << "  " << prettyString(s.nodeStart, "  ", false) << '\n'
                  << "should look like:\n"
                  << "  for(int var = off; var [<] dim; ...; ...)\n"
                  << "                          ^ less than\n";
        throw 1;
      }

      nodePos = nodePos->right;

      // Dim is shadowed by a variable
      if((nodePos->type & unknownVariable) &&
         (nodePos->right == commaNodes[2])){

        dimName = nodePos->value;
      }
      // Dim explicitly stated
      else{
        bool startFromBelow = false;

        // blah < [(...) ...]
        if(nodePos->left->down.size()){
          nodePos = nodePos->left;
          startFromBelow = true;
        }

        // Un-link node chain
        commaNodes[2]->left->right = NULL;

        // Manual bounds
        bool errorFound = nodeHasUnknownVariable(nodePos);

        if(!errorFound){
          strNode *tmpNodePos;

          if(startFromBelow){
            tmpNodePos = nodePos->clone();

            tmpNodePos->value = "+";
            tmpNodePos->type  = keywordType["+"];
          }
          else
            tmpNodePos = nodePos;

          dimSubName = (std::string) evaluateLabelNode(tmpNodePos);

          // Re-link node chain
          commaNodes[2]->left->right = commaNodes[2]->left;
        }
        else{
          std::cout << "The second statement of:\n"
                    << "  " << prettyString(s.nodeStart, "  ", false) << '\n'
                    << "should look like:\n"
                    << "  for(int var = off; var < [dim]; ...; ...)\n"
                    << "                            ^ only one variable name\n"
                    << "                              or exact value\n";
          throw 1;
        }
      }

      if(dimName.size())
        dimVar.name = dimName;
      else
        dimVar.name = obfuscate("occa" + ioLoop + "Dim" + loopNest);

      if(dimSubName.size())
        dimVar.extraInfo.push_back(dimSubName);
      else
        dimVar.extraInfo.push_back("occa" + ioLoop + "Dim" + loopNest);

      // [-] ?
      if(!sKernel.hasVariableInScope(dimVar.name))
        sKernel.addVariable(dimVar);
      s.addVariable(dimVar);

      nodePos = commaNodes[3]->right;

      loopVar.name = "occa" + ioLoop + "For" + loopNest;

      s.addVariable(loopVar);

      std::string occaForName = "occa" + ioLoop + "For" + loopNest;

      // [-] Needs proper free (can't because it's nested...)
      // free(s.nodeStart);

      s.type            = keywordType["occaOuterFor0"];
      s.nodeStart       = new strNode(occaForName);
      s.nodeStart->type = s.type;

      s.nodeEnd = s.nodeStart;

      const int downCount = oldDown.size();

      for(int i = 0; i < downCount; ++i){
        // Remove the braces
        popAndGoRight(oldDown[i]);
        parserNamespace::lastNode(oldDown[i])->pop();

        s.loadAllFromNode(oldDown[i]);
      }
    }

    inline void parserBase::loadScopeVarMap(statement &s){
      if((!(s.type & declareStatementType)   &&
          !(s.type & forStatementType)       &&
          !(s.type & gotoStatementType)      &&
          !(s.type & functionStatementType)) ||
         // OCCA for's don't have arguments
         (s.type == (forStatementType | occaStatementType)))
        return;

      strNode *nodePos = s.nodeStart;

      statement *up = s.up;

      if(s.type & functionStatementType){
        varInfo info = loadVarInfo(nodePos);
        (s.up)->addVariable(info, &s);
      }

      if(s.type & (forStatementType |
                   functionStatementType)){
        while(nodePos->down.size() == 0)
          nodePos = nodePos->right;

        nodePos = nodePos->down[0]->right;

        up = &s;
      }

      if( !(s.type & functionPrototypeType) ){
        varInfo info = loadVarInfo(nodePos);

        if(info.typeInfo & functionCallType)
          return;

        up->addVariable(info, &s);

        while(nodePos){
          if(nodePos->value == ","){
            nodePos = nodePos->right;

            varInfo info2 = loadVarInfo(nodePos);

            // Functions have types for each argument
            if( !(s.type & functionStatementType) ){
              info2.type        = info.type;
              info2.descriptors = info.descriptors;
            }

            up->addVariable(info2, &s);
          }
          else
            nodePos = nodePos->right;
        }
      }
    }

    inline bool parserBase::statementHasOccaFor(statement &s){
      if(s.type == keywordType["occaOuterFor0"])
        return true;

      statementNode *statementPos = s.statementStart;

      while(statementPos){
        if( statementHasOccaFor(*(statementPos->value)) )
          return true;

        statementPos = statementPos->right;
      }

      return false;
    }

    inline bool parserBase::statementHasOklFor(statement &s){
      if(s.type == forStatementType){

        strNode *nodePos = s.nodeStart;

        while(nodePos->down.size() == 0)
          nodePos = nodePos->right;

        nodePos = nodePos->down[0];

        while(nodePos){
          if( ((nodePos->value.find("inner") != std::string::npos) &&
               ((nodePos->value == "inner0") ||
                (nodePos->value == "inner1") ||
                (nodePos->value == "inner2")))                     ||
              ((nodePos->value.find("outer") != std::string::npos) &&
               ((nodePos->value == "outer0") ||
                (nodePos->value == "outer1") ||
                (nodePos->value == "outer2"))) ){

            return true;
          }

          nodePos = nodePos->right;
        }
      }

      statementNode *statementPos = s.statementStart;

      while(statementPos){
        if( statementHasOklFor(*(statementPos->value)) )
          return true;

        statementPos = statementPos->right;
      }

      return false;
    }

    inline void parserBase::labelKernelsAsNativeOrNot(statement &s){
      if(!statementIsAKernel(s))
        return;

      bool hasOccaFor = statementHasOccaFor(s);
      bool hasOklFor  = statementHasOklFor(s);

      if(hasOccaFor | hasOklFor){
        varInfo nativeCheckVar;

        if(hasOccaFor)
          nativeCheckVar.name = obfuscate("native", "occa");
        else
          nativeCheckVar.name = obfuscate("native", "okl");

        varInfo *nativeCheckVar2 = s.hasVariableInScope(nativeCheckVar.name);

        if(nativeCheckVar2 == NULL)
          s.addVariable(nativeCheckVar);
      }
    }

    inline void parserBase::setupCudaVariables(statement &s){
      if((!(s.type & declareStatementType)   &&
          !(s.type & forStatementType)       &&
          !(s.type & functionStatementType)) ||
         // OCCA for's don't have arguments
         (s.type == keywordType["occaOuterFor0"]))
        return;

      if(getStatementKernel(s) == NULL)
        return;

      if(statementKernelUsesNativeLanguage(s))
        return;

      strNode *nodePos = s.nodeStart;

      statement *up = s.up;

      // Go to [(]
      if(s.type & functionStatementType)
        loadVarInfo(nodePos);

      if(s.type & (forStatementType |
                   functionStatementType)){
        while(nodePos->down.size() == 0)
          nodePos = nodePos->right;

        nodePos = nodePos->down[0]->right;

        up = &s;
      }

      if( !(s.type & functionPrototypeType) ){
        while(nodePos){
          if((nodePos->type & cudaKeywordType) &&
             (nodePos->right)                 &&
             (nodePos->right->value == ".")  &&
             (nodePos->right->right)){

            std::string &coord = nodePos->right->right->value;

            if((coord.size() == 1) &&
               ('x' <= coord[0]) && (coord[0] <= 'z')){
              std::string occaCoord = coord.substr(0,1);
              occaCoord[0] += ('0' - 'x');

              bool compressing = false;

              if(nodePos->value == "threadIdx"){
                compressing = true;
                nodePos->value = "occaInnerId" + occaCoord;
              }
              else if(nodePos->value == "blockDim"){
                compressing = true;
                nodePos->value = "occaInnerDim" + occaCoord;
              }
              else if(nodePos->value == "blockIdx"){
                compressing = true;
                nodePos->value = "occaOuterId" + occaCoord;
              }
              else if(nodePos->value == "gridDim"){
                compressing = true;
                nodePos->value = "occaOuterDim" + occaCoord;
              }

              if(compressing){
                nodePos->type = keywordType[nodePos->value];
                nodePos->right->pop();
                nodePos->right->pop();
              }
            }
          }

          if(nodePos != NULL)
            nodePos = nodePos->right;
        }
      }
    }

    inline void parserBase::loadVariableInformation(statement &s,
                                                    strNode *n){
      if(s.type & functionPrototypeType)
        return;

      while(n){
        if(n->type & unknownVariable){
          varInfo *infoPtr = s.hasVariableInScope(n->value);

          if(infoPtr == NULL){
            std::cout << "Couldn't find [" << (std::string) *n << "] in:\n"
                      << s << '\n';
            throw 1;
          }

          varUsedMap[infoPtr].push(&s);
        }

        const int downCount = n->down.size();

        for(int i = 0; i < downCount; ++i)
          loadVariableInformation(s, n->down[i]);

        n = n->right;
      }
    }

    inline void parserBase::loadVariableInformation(statement &s){
      loadScopeVarMap(s);

      loadVariableInformation(s, s.nodeStart);
    }

    inline void parserBase::addFunctionPrototypes(statement &s){
      // Global scope only
      if(0 <= s.depth)
        return;

      std::map<std::string,bool> prototypes;

      statementNode *statementPos = s.statementStart;

      while(statementPos){
        statement *s2 = statementPos->value;

        if(s2->type & functionPrototypeType){
          strNode *nodePos = s2->nodeStart;
          varInfo info = loadVarInfo(nodePos);

          prototypes[info.name] = true;
        }

        statementPos = statementPos->right;
      }

      statementPos = s.statementStart;

      while(statementPos){
        statement *s2 = statementPos->value;

        if(s2->type & functionStatementType){
          strNode *nodePos = s2->nodeStart;
          varInfo info = loadVarInfo(nodePos);

          if(info.hasDescriptor("occaKernel") ||
             info.hasDescriptor("kernel")){

            statementPos = statementPos->right;
            continue;
          }

          if(!info.hasDescriptor("occaFunction")){
            strNode *ofNode = s2->nodeStart;

            ofNode       = ofNode->push("occaFunction");
            ofNode->type = keywordType["occaFunction"];

            ofNode->swapWithLeft();
            s2->nodeStart = ofNode;
          }

          if( !(s2->type & functionDefinitionType) ){
            statementPos = statementPos->right;
            continue;
          }

          if(prototypes.find(info.name) == prototypes.end()){
            statement *newS2 = s2->clone();
            statementNode *newNode = new statementNode(newS2);

            newS2->type = functionPrototypeType;

            newS2->statementCount = 0;

            // [-] Delete definition (needs proper free)
            delete newS2->statementStart;
            newS2->statementStart = NULL;
            newS2->statementEnd   = NULL;

            strNode *end = newS2->nodeEnd;
            end = end->push(";");

            end->up        = newS2->nodeEnd->up;
            end->type      = keywordType[";"];
            end->depth     = newS2->nodeEnd->depth;
            end->sideDepth = newS2->nodeEnd->sideDepth;

            newS2->nodeEnd = end;

            statementNode *left = statementPos->left;

            newNode->left = left;
            if(left)
              left->right = newNode;

            newNode->right     = statementPos;
            statementPos->left = newNode;

            ++(s.statementCount);
          }
        }

        statementPos = statementPos->right;
      }
    }

    inline int parserBase::statementOccaForNest(statement &s){
      if( !(s.type & (forStatementType | occaStatementType)) )
        return notAnOccaFor;

      int ret = notAnOccaFor;

      const std::string &forName = s.nodeStart->value;

      if((forName.find("occaOuterFor") != std::string::npos) &&
         ((forName == "occaOuterFor0") ||
          (forName == "occaOuterFor1") ||
          (forName == "occaOuterFor2"))){

        ret = ((1 + forName[12] - '0') << occaOuterForShift);
      }
      else if((forName.find("occaInnerFor") != std::string::npos) &&
              ((forName == "occaInnerFor0") ||
               (forName == "occaInnerFor1") ||
               (forName == "occaInnerFor2"))){

        ret = ((1 + forName[12] - '0') << occaInnerForShift);
      }

      return ret;
    }

    inline bool parserBase::statementIsAnOccaFor(statement &s){
      const int nest = statementOccaForNest(s);

      return !(nest & notAnOccaFor);
    }

    inline void parserBase::fixOccaForStatementOrder(statement &origin,
                                                     statementNode *sn){
      int innerLoopCount = -1;

      while(sn){
        std::vector<statement*> statStack;
        std::vector<int> nestStack;

        statement &s = *(sn->value);

        const int sNest = statementOccaForNest(s);

        if(sNest & notAnOccaFor){
          sn = sn->right;
          continue;
        }

        const bool isAnInnerLoop = (sNest & occaInnerForMask);

        const int shift = (isAnInnerLoop ? occaInnerForShift : occaOuterForShift);
        const int mask  = (isAnInnerLoop ? occaInnerForMask  : occaOuterForMask);

        statement *sp = &s;

        statStack.push_back(sp);
        nestStack.push_back((sNest >> shift) - 1);

        int loopCount = 1;

        sp = sp->statementStart->value;

        while(sp){
          const int nest = statementOccaForNest(*sp);

          if(nest & (~mask))
            break;

          statStack.push_back(sp);
          nestStack.push_back((nest >> shift) - 1);

          ++loopCount;

          sp = sp->statementStart->value;
        }

        if(isAnInnerLoop){
          if(innerLoopCount == -1){
            innerLoopCount = loopCount;
          }
          else{
            if(loopCount != innerLoopCount){
              std::cout << "Inner loops are inconsistent in:\n"
                        << origin << '\n';

              throw 1;
            }

            if(!statementHasBarrier( *(sn->left->value) )){
              std::cout << "Warning: Placing a local barrier between:\n"
                        << "---[ A ]--------------------------------\n"
                        << *(sn->left->value)
                        << "---[ B ]--------------------------------\n"
                        << *(sn->value)
                        << "========================================\n";

              statement *newS = new statement(sn->value->depth,
                                              declareStatementType, sn->value->up,
                                              NULL, NULL);

              statementNode *newSN = new statementNode(newS);

              newS->nodeStart = splitFileContents("occaBarrier(occaLocalMemFence);\0");
              newS->nodeStart = labelCode(newS->nodeStart);
              newS->nodeEnd   = lastNode(newS->nodeStart);

              if(sn->left)
                sn->left->right = newSN;

              newSN->left  = sn->left;
              newSN->right = sn;

              sn->left = newSN;
            }
          }
        }
        else{
          // Re-order inner loops
          fixOccaForStatementOrder(origin, (sp->up->statementStart));
        }

        std::sort(nestStack.begin(), nestStack.end());

        for(int i = (loopCount - 1); 0 <= i; --i){
          if(nestStack[i] != i){
            std::cout << "Inner loops ";

            for(int i2 = 0; i2 < loopCount; ++i2)
              std::cout << (i2 ? ", " : "[") << nestStack[i2];

            std::cout << "] have duplicates or gaps:\n"
                      << origin << '\n';

            throw 1;
          }

          sp = statStack[loopCount - i - 1];

          sp->nodeStart->value  = "occa";
          sp->nodeStart->value += (isAnInnerLoop ? "Inner" : "Outer");
          sp->nodeStart->value += "For0";

          sp->nodeStart->value[12] += i;
        }

        sn = sn->right;
      }
    }

    inline void parserBase::fixOccaForOrder(statement &s){
      if( !statementIsAKernel(s) )
        return;

      fixOccaForStatementOrder(s, s.statementStart);
    }

    inline void parserBase::addParallelFors(statement &s){
      if( !statementIsAKernel(s) )
        return;

      statementNode *snPos = s.statementStart;

      while(snPos){
        statement &s2 = *(snPos->value);

        const int nest = statementOccaForNest(s2);

        if(nest & (notAnOccaFor | occaInnerForMask)){

          snPos = snPos->right;
          continue;
        }

        const char outerDim = '0' + (nest - 1);

        statement *parallelStatement = new statement(s.depth + 1,
                                                     occaStatementType, &s,
                                                     NULL, NULL);

        statementNode *parallelSN = new statementNode(parallelStatement);

        parallelStatement->nodeStart         = new strNode("occaParallelFor");
        parallelStatement->nodeStart->value += outerDim;
        parallelStatement->nodeStart->value += '\n';
        parallelStatement->type              = occaStatementType;

        if(s.statementStart == snPos)
          s.statementStart = parallelSN;

        statementNode *leftSN  = snPos->left;

        parallelSN->right = snPos;
        parallelSN->left  = leftSN;

        snPos->left = parallelSN->right;

        if(leftSN)
          leftSN->right = parallelSN;

        snPos = snPos->right;
      }
    }

    inline void parserBase::updateConstToConstant(statement &s){
      // Global scope only
      if((s.depth != 0) ||
         !(s.type & declareStatementType))
        return;

      strNode *nodePos = s.nodeStart;

      while(nodePos){
        if(nodePos->value == "occaConst")
          nodePos->value = "occaConstant";

        // [*] or [&]
        if(nodePos->type & (unknownVariable |
                            binaryOperatorType))
          break;

        nodePos = nodePos->right;
      }
    }

    inline strNode* parserBase::occaExclusiveStrNode(varInfo &info,
                                                     const int depth,
                                                     const int sideDepth){
      strNode *nodeRoot;

      const int typeInfo = info.typeInfo;

      if(typeInfo & pointerType)
        nodeRoot = new strNode("occaPrivateArray");
      else
        nodeRoot = new strNode("occaPrivate");

      nodeRoot->type      = presetValue;
      nodeRoot->depth     = depth;
      nodeRoot->sideDepth = sideDepth;

      strNode *nodePos = nodeRoot->pushDown("(");

      nodePos->type  = keywordType["("];
      nodePos->depth = depth + 1;

      const int descriptorCount = info.descriptors.size();

      for(int i = 0; i < descriptorCount; ++i){
        if(info.descriptors[i] == "exclusive")
          continue;

        nodePos       = nodePos->push(info.descriptors[i]);
        nodePos->type = qualifierType;
      }

      if(info.type.size()){
        std::string infoType = info.type;

        nodePos       = nodePos->push(infoType);
        nodePos->type = specifierType;

        if(typeInfo & heapPointerType){
          for(int i = 0; i < info.pointerCount; ++i){
            nodePos       = nodePos->push("*");
            nodePos->type = keywordType["*"];
          }
        }
      }

      nodePos       = nodePos->push(",");
      nodePos->type = keywordType[","];

      nodePos       = nodePos->push(info.name);
      nodePos->type = unknownVariable;

      if(typeInfo & stackPointerType){
        const int heapCount = info.stackPointerSizes.size();

        if(1 < heapCount){
          std::cout << "Only 1D exclusive variables are currently supported [" << info << "]\n";
          throw 1;
        }

        nodePos       = nodePos->push(",");
        nodePos->type = keywordType[","];

        nodePos       = nodePos->push(info.stackPointerSizes[0]);
        nodePos->type = presetValue;
      }

      nodePos       = nodePos->push(")");
      nodePos->type = keywordType[")"];

      nodePos       = nodePos->push(";");
      nodePos->type = keywordType[";"];

      return nodeRoot;
    }

    inline void parserBase::addKernelInfo(varInfo &info, statement &s){
      if( !(s.type & functionStatementType) ){
        node<strNode*> nNodeRoot = s.nodeStart->getStrNodesWith(info.name);

        node<strNode*> *nNodePos = nNodeRoot.right;

        while(nNodePos){
          strNode *nodePos = nNodePos->value;

          // If we're calling function, not using a function pointer
          if(nodePos->down.size()){
            nodePos = nodePos->down[0];

            if((nodePos->type == startParentheses) &&
               (nodePos->value != "occaKernelInfo")){
              strNode *kia = nodePos->push("occaKernelInfo");

              kia->up        = nodePos->up;
              kia->type      = keywordType["occaKernelInfo"];
              kia->depth     = nodePos->depth;
              kia->sideDepth = nodePos->sideDepth;

              strNode *comma = kia->push(",");

              comma->up        = kia->up;
              comma->type      = keywordType[","];
              comma->depth     = kia->depth;
              comma->sideDepth = kia->sideDepth;
            }
          }

          nNodePos = nNodePos->right;
        }
      }
    }

    inline void parserBase::addArgQualifiers(varInfo &info, statement &s){
      // Having functionCallType at this level means:
      //   occaExp, occaBarrier, etc
      // so don't add occaKernelInfoArg
      if((info.typeInfo & functionType) &&
         !(info.typeInfo & functionCallType)){
        strNode *nodePos = s.nodeStart;

        while(nodePos->down.size() == 0)
          nodePos = nodePos->right;

        nodePos = nodePos->down[0];

        if((nodePos->type == startParentheses) &&
           (nodePos->value != "occaKernelInfoArg")){
          strNode *kia = nodePos->push("occaKernelInfoArg");

          kia->up        = nodePos->up;
          kia->type      = keywordType["occaKernelInfoArg"];
          kia->depth     = nodePos->depth;
          kia->sideDepth = nodePos->sideDepth;

          strNode *comma = kia->push(",");

          comma->up        = kia->up;
          comma->type      = keywordType[","];
          comma->depth     = kia->depth;
          comma->sideDepth = kia->sideDepth;

          applyToStatementsUsingVar(info, &parserBase::addKernelInfo);

          while(nodePos){
            if((nodePos->value == ",") ||
               (nodePos->value == ")"))
              break;

            nodePos = nodePos->right;
          }
        }

        if(!info.hasDescriptor("occaKernel") &&
           !info.hasDescriptor("kernel"))
          return;

        while(nodePos){
          if(nodePos->value == ","){
            nodePos = nodePos->right;

            strNode *nextVar = nodePos;
            varInfo info = loadVarInfo(nextVar);

            if((info.typeInfo & pointerType)        &&
               (!info.hasDescriptor("occaPointer")) &&
               (!info.hasDescriptor("texture")) ){
              nodePos       = nodePos->push("occaPointer");
              nodePos->type = keywordType["occaPointer"];

              nodePos->swapWithLeft();
            }
            else if(!(info.typeInfo & pointerType) &&
                    !info.hasDescriptor("occaVariable") ){
              while( !(nodePos->type & unknownVariable) )
                nodePos = nodePos->right;

              nodePos       = nodePos->push("occaVariable");
              nodePos->type = keywordType["occaVariable"];

              nodePos->swapWithLeft();
            }

            nodePos = nextVar;
          }
          else
            nodePos = nodePos->right;
        }
      }
    }

    inline void parserBase::modifyExclusiveVariables(statement &s){
      if(!(s.type & declareStatementType))
        return;

      if(getStatementKernel(s) == NULL)
        return;

      if(statementKernelUsesNativeOCCA(s))
        return;

      strNode *nodePos = s.nodeStart;
      varInfo info = loadVarInfo(nodePos);

      if(!info.hasDescriptor("exclusive"))
        return;

      statement *newS = new statement(s.depth,
                                      declareStatementType, s.up,
                                      NULL, NULL);

      newS->nodeStart = occaExclusiveStrNode(info,
                                             nodePos->depth,
                                             nodePos->sideDepth);

      newS->nodeEnd   = lastNode(newS->nodeStart);

      statementNode *newSN = new statementNode(newS);
      statementNode *upPos = (s.up)->statementStart;

      while(upPos->value != &s)
        upPos = upPos->right;

      statementNode *oldSN = upPos;

      if(upPos == (s.up)->statementStart){
        (s.up)->statementStart = newSN;

        if(oldSN->right)
          oldSN->right->left = newSN;

        newSN->left  = NULL;
        newSN->right = oldSN->right;
      }
      else{
        oldSN->left->right = newSN;

        if(oldSN->right)
          oldSN->right->left = newSN;

        newSN->left  = oldSN->left;
        newSN->right = oldSN->right;
      }

      while(nodePos){
        if(nodePos->value == ","){
          nodePos = nodePos->right;

          varInfo info2 = loadVarInfo(nodePos);

          info2.type        = info.type;
          info2.descriptors = info.descriptors;

          statement *newS2 = new statement(s.depth,
                                           declareStatementType, s.up,
                                           NULL, NULL);

          newS2->nodeStart = occaExclusiveStrNode(info2,
                                                  nodePos->depth,
                                                  nodePos->sideDepth);

          newS2->nodeEnd   = lastNode(newS2->nodeStart);

          newSN = newSN->push(newS2);
        }
        else
          nodePos = nodePos->right;
      }

      // [-] Needs proper free (can't because it's nested...)
      //   delete oldSN;
    }

    inline void parserBase::modifyStatementOccaForVariables(varInfo &var,
                                                            strNode *n){
      const int extras = var.extraInfo.size();

      while(n){
        if((n->type & unknownVariable) &&
           (n->value == var.name)){

          if(extras == 1)
            n->value = var.extraInfo[0];
          else
            n->value = ("(" + (var.extraInfo[0] +
                               " + " +
                               var.extraInfo[1]) +
                        ")");
        }

        const int downCount = n->down.size();

        for(int i = 0; i < downCount; ++i)
          modifyStatementOccaForVariables(var, n->down[i]);

        n = n->right;
      }
    }

    inline void parserBase::modifyOccaForVariables(){
      varUsedMapIterator it = varUsedMap.begin();

      while(it != varUsedMap.end()){
        varInfo *infoPtr = it->first;
        const int extras = infoPtr->extraInfo.size();

        // Only occa id/dim's have extras atm
        if(extras){
          // First node is just a placeholder
          statementNode *sNodePos = (it->second).right;

          while(sNodePos){
            statement &s = *(sNodePos->value);

            modifyStatementOccaForVariables(*infoPtr,
                                            s.nodeStart);

            sNodePos = sNodePos->right;
          }
        }

        ++it;
      }
    }

    inline void parserBase::modifyTextureVariables(){
      varUsedMapIterator it = varUsedMap.begin();

      while(it != varUsedMap.end()){
        varInfo *infoPtr = it->first;

        if(infoPtr->hasDescriptor("texture")){
          statement &os = *(varOriginMap[infoPtr]);

          strNode *osNodePos = os.nodeStart;

          while(osNodePos){
            if(osNodePos->value == infoPtr->name)
              std::cout << "HERE!\n";

            osNodePos = osNodePos->right;
          }

          // First node is just a placeholder
          statementNode *sNodePos = (it->second).right;

          while(sNodePos){
            statement &s = *(sNodePos->value);
            strNode *nodePos = s.nodeStart;

            while(nodePos){
              if((nodePos->type & unknownVariable) &&
                 (nodePos->value == infoPtr->name)){
                // [-] HERE
                std::cout
                  << "2. nodePos = " << *nodePos << '\n';
              }

              nodePos = nodePos->right;
            }

            sNodePos = sNodePos->right;
          }
        }

        ++it;
      }
    }

    inline std::string parserBase::occaScope(statement &s){
      statement *currentS = &s;

      while(currentS){
        if(currentS->type & (forStatementType | occaStatementType))
          break;

        currentS = currentS->up;
      }

      if(currentS == NULL)
        return "";

      return s.nodeStart->value;
    }

    inline void parserBase::incrementDepth(statement &s){
      ++s.depth;
    }

    inline void parserBase::decrementDepth(statement &s){
      --s.depth;
    }

    inline bool parserBase::statementHasBarrier(statement &s){
      strNode *n = s.nodeStart;

      while(n){
        if(n->value == "occaBarrier")
          return true;

        n = n->right;
      }

      return false;
    }

    inline statementNode* parserBase::findStatementWith(statement &s,
                                                        findStatementWith_t func){
      statementNode *ret = new statementNode(&s);

      if((this->*func)(s))
        return ret;

      statementNode *statementPos = s.statementStart;

      int found = 0;

      while(statementPos){
        statementNode *ret2 = findStatementWith(*(statementPos->value), func);

        if(ret2 != NULL){
          ret->pushDown(ret2);
          ++found;
        }

        statementPos = statementPos->right;
      }

      if(found)
        return ret;

      delete ret;

      return NULL;
    }

    inline int parserBase::getKernelOuterDim(statement &s){
      statementNode *statementPos = s.statementStart;

      std::string outerStr = obfuscate("Outer");
      int outerDim = -1;

      varInfo *info = s.hasVariableInScope(outerStr);

      if(info != NULL){
        const int extras = info->extraInfo.size();

        for(int i = 0; i < extras; ++i){
          const int loopNest = (info->extraInfo[i][0] - '0');

          if(outerDim < loopNest)
            outerDim = loopNest;

          // Max Dim
          if(outerDim == 2)
            return outerDim;
        }
      }

      while(statementPos){
        const int outerDim2 = getKernelOuterDim( *(statementPos->value) );

        if(outerDim < outerDim2)
          outerDim = outerDim2;

        // Max Dim
        if(outerDim == 2)
          return outerDim;

        statementPos = statementPos->right;
      }

      return outerDim;
    }

    inline int parserBase::getKernelInnerDim(statement &s){
      statementNode *statementPos = s.statementStart;

      std::string innerStr = obfuscate("Inner");
      int innerDim = -1;

      varInfo *info = s.hasVariableInScope(innerStr);

      if(info != NULL){
        const int extras = info->extraInfo.size();

        for(int i = 0; i < extras; ++i){
          const int loopNest = (info->extraInfo[i][0] - '0');

          if(innerDim < loopNest)
            innerDim = loopNest;

          // Max Dim
          if(innerDim == 2)
            return innerDim;
        }
      }

      while(statementPos){
        const int innerDim2 = getKernelInnerDim( *(statementPos->value) );

        if(innerDim < innerDim2)
          innerDim = innerDim2;

        // Max Dim
        if(innerDim == 2)
          return innerDim;

        statementPos = statementPos->right;
      }

      return innerDim;
    }

    inline void parserBase::checkPathForConditionals(statementNode *path){
      if((path == NULL) ||
         (path->value == NULL))
        return;

      if(path->value->type & ifStatementType){
        std::cout << '\n'
                  << "+---------+--------------------------------------------------+\n"
                  << "|         | Barriers inside conditional statements will only |\n"
                  << "| Warning | work properly if it's always called or never     |\n"
                  << "|         | called.                                          |\n"
                  << "+---------+--------------------------------------------------+\n"
                  << *(path->value)
                  << "+---------+--------------------------------------------------+\n\n";
      }

      const int downCount = path->down.size();

      for(int i = 0; i < downCount; ++i)
        checkPathForConditionals(path->down[i]);
    }

    inline int parserBase::findLoopSections(statement &s,
                                            statementNode *path,
                                            loopSection_t &loopSection,
                                            int section){
      if(s.statementCount == 0)
        return section;

      int downCount = 0;
      int downPos   = 0;

      statementNode *sPos = s.statementStart;
      statement *sNext = NULL;

      if(path != NULL){
        downCount = path->down.size();

        if(downCount)
          sNext = path->down[0]->value;
      }

      while(sPos){
        if(sPos->value == sNext){
          // Last one is a barrier
          if(path->down[downPos]->down.size() == 0)
            ++section;

          section = findLoopSections(*(sPos->value),
                                     path->down[downPos],
                                     loopSection,
                                     section);

          if(++downPos < downCount)
            sNext = path->down[downPos]->value;
          else
            sNext = NULL;
        }
        else{
          loopSection[sPos->value] = section;

          findLoopSections(*(sPos->value),
                           NULL,
                           loopSection,
                           section);
        }

        sPos = sPos->right;
      }

      return section;
    }

    inline bool parserBase::varInTwoSegments(varInfo &info,
                                             loopSection_t &loopSection){
      varUsedMapIterator it = varUsedMap.find(&info);

      // Variable is not used
      if(it == varUsedMap.end())
        return false;

      statementNode *pos = (it->second).right;

      const int segment = loopSection[pos->value];
      pos = pos->right;

      while(pos){
        if(segment != loopSection[pos->value])
          return true;

        pos = pos->right;
      }

      return false;
    }

    inline varInfoNode* parserBase::findVarsMovingToTop(statement &s,
                                                        loopSection_t &loopSection){
      // Statement defines have doubles (to know how many variables
      //                                 were defined)
      //    so ... ignore duplicates
      const bool ignoringVariables = (s.type & simpleStatementType);

      scopeVarMapIterator it = s.scopeVarMap.begin();
      varInfoNode *root = NULL;
      varInfoNode *pos  = NULL;

      if(!ignoringVariables){
        while(it != s.scopeVarMap.end()){
          varInfo &info = *(it->second);

          if(info.hasDescriptor("occaShared") ||
             varInTwoSegments(info, loopSection)){
            if(root == NULL){
              root = new varInfoNode(&info);
              pos  = root;
            }
            else
              pos = pos->push(&info);
          }

          ++it;
        }
      }

      statementNode *sn = s.statementStart;

      while(sn){
        varInfoNode *pos2 = findVarsMovingToTop(*(sn->value),
                                                loopSection);

        if(pos2 != NULL){
          pos->right = pos2;
          pos2->left = pos;

          pos = lastNode(pos);
        }

        sn = sn->right;
      }

      return root;
    }

    inline void parserBase::splitDefineForVariable(statement *&origin,
                                                   varInfo &var, strNode *varNode,
                                                   const int declPos){
      const int declarationCount = (origin->scopeVarMap.size() == 1);
      const bool addingStatement = !((declPos == 0) &&
                                     (declarationCount == 1));

      statement &originUp  = *(origin->up);
      statementNode *snPos = NULL;

      bool removeVarStatement = true;
      bool ignoringFirst      = false;
      bool ignoringSecond     = false;

      if(addingStatement){
        snPos = originUp.statementStart;

        while(snPos->value != origin)
          snPos = snPos->right;

        // If it's something like
        //   const int [a = 0], b = 0 ...
        // stitch
        //   const int b = 0 ...
        // and paste
        //   a = 0;
        if(varNode->right->value == "="){
          removeVarStatement = false;
          varNode = varNode->left;
        }
      }

      if(declPos == 0){
        // const int [a = 0];
        if(declarationCount == 1){
          while(origin->nodeStart != varNode)
            popAndGoRight(origin->nodeStart);

          origin->type = updateStatementType;
        }
        // const int [a = 0], ...;
        else{
          ignoringSecond = true;

          // Removing const int [* const] a = NULL
          while(varNode->type != specifierType)
            popAndGoLeft(varNode);

          varNode = varNode->right;

          if(removeVarStatement){
            ignoringFirst = true;

            while(varNode->value != ",")
              popAndGoRight(varNode);
          }
          else{
            while(varNode->value != ",")
              varNode = varNode->right;
          }

          // Remove the right [,]
          popAndGoRight(varNode);
        }
      }
      // const int a = 0, [b = 0], ...;
      else{
        while(varNode->value != ",")
          popAndGoLeft(varNode);

        if(removeVarStatement){
          while((varNode->value != ",") &&
                (varNode->value != ";"))
            popAndGoRight(varNode);
        }
        else{
          while((varNode->value != ",") &&
                (varNode->value != ";"))
            varNode = varNode->right;
        }

        // Remove the right [,]
        popAndGoRight(varNode);
      }

      origin->scopeVarMap.erase(var.name);

      var.descriptors.insert(var.descriptors.begin(), "exclusive");
      strNode *newVarNode = var.makeStrNodeChain();

      statement *newS = new statement(origin->depth,
                                      declareStatementType,
                                      origin->up,
                                      newVarNode, lastNode(newVarNode));

      varOriginMap[&var]          = newS;
      newS->scopeVarMap[var.name] = &var;

      statementNode *newVarSN = new statementNode(newS);

      statement *rootStatement = origin;

      while( !(rootStatement->type & (forStatementType |
                                      functionStatementType)) )
        rootStatement = rootStatement->up;

      statementNode *oldFirstPos = rootStatement->statementStart;

      rootStatement->statementStart = newVarSN;
      newVarSN->right               = oldFirstPos;
      oldFirstPos->left             = newVarSN;

      newS->depth = rootStatement->depth + 1;
      newS->up    = rootStatement;

      if(!addingStatement)
        return;

      varUsedMap[&var].push(origin);

      strNode *firstNodeStart = firstNode(varNode);
      strNode *firstNodeEnd    = varNode->left;
      strNode *secondNodeStart = varNode;

      secondNodeStart->left = NULL;

      if(firstNodeEnd){
        firstNodeEnd->right = NULL;
        firstNodeEnd        = firstNodeEnd->push(";");
        firstNodeEnd->type  = keywordType[";"];

        // Seal off first define
        origin->nodeEnd = firstNodeEnd;
      }

      strNode *secondNodeEnd = secondNodeStart;

      if(ignoringSecond){
        while(secondNodeEnd->value != ";")
          secondNodeEnd = secondNodeEnd->right;
      }
      else{
        while((secondNodeEnd->value != ",") &&
              (secondNodeEnd->value != ";"))
          secondNodeEnd = secondNodeEnd->right;
      }

      secondNodeEnd->value = ";";
      secondNodeEnd->type  = keywordType[";"];

      // Create second update
      if(!ignoringSecond &&
         !removeVarStatement){
        statement *secondS = new statement(origin->depth,
                                           updateStatementType,
                                           origin->up,
                                           secondNodeStart, secondNodeEnd);

        snPos = snPos->push(secondS);
        varUsedMap[&var].push(secondS);
      }

      // Create third define
      strNode *thirdNodeStart = ignoringSecond ? secondNodeStart : secondNodeEnd->right;
      strNode *thirdNodeEnd   = lastNode(thirdNodeStart);

      if(thirdNodeStart){
        secondNodeEnd->right = NULL;
        thirdNodeStart->left = NULL;

        // Copy over the desciptors to the next statement
        strNode *thirdPrefix = firstNodeStart->clone();

        if( !(thirdPrefix->type & specifierType) ){
          strNode *firstNodePos = firstNodeStart->right;

          while( !(firstNodePos->type & specifierType) ){
            thirdPrefix = thirdPrefix->push( firstNodePos->clone() );
            firstNodePos = firstNodePos->right;
          }
        }

        thirdPrefix->right   = thirdNodeStart;
        thirdNodeStart->left = thirdPrefix;

        thirdNodeStart = firstNode(thirdNodeStart);

        statement *thirdS;

        if(!ignoringFirst){
          thirdS = new statement(origin->depth,
                                 declareStatementType,
                                 origin->up,
                                 thirdNodeStart, thirdNodeEnd);

          snPos = snPos->push(thirdS);
        }
        else
          thirdS = origin;

        thirdNodeEnd = thirdNodeStart;

        while(thirdNodeEnd){
          if(thirdNodeEnd->type & unknownVariable){
            scopeVarMapIterator it = origin->scopeVarMap.find(thirdNodeEnd->value);
            varInfo &movingVar = *(it->second);

            origin->scopeVarMap.erase(it);

            varOriginMap[&movingVar]            = thirdS;
            thirdS->scopeVarMap[movingVar.name] = &movingVar;

            while((thirdNodeEnd->value != ",") &&
                  (thirdNodeEnd->value != ";"))
              thirdNodeEnd = thirdNodeEnd->right;
          }

          thirdNodeEnd = thirdNodeEnd->right;
        }
      }

      if(!ignoringFirst){
        // Gotta remove the descriptors
        if(ignoringSecond){
          while( !(firstNodeStart->type & unknownVariable) )
            popAndGoRight(firstNodeStart);
        }

        origin->nodeStart = firstNodeStart;
        origin->nodeEnd   = firstNodeEnd;
      }
      else{
        origin->nodeStart = thirdNodeStart;
        origin->nodeEnd   = thirdNodeEnd;
      }
    }

    inline void parserBase::addInnerForsBetweenBarriers(statement &origin,
                                                        statementNode *s,
                                                        const int innerDim){
      const int occaForType = keywordType["occaInnerFor0"];

      statement *outerMostLoop = NULL;
      statement *innerMostLoop = NULL;

      while(s &&
            statementHasBarrier( *(s->value) ))
        s = s->right;

      while(s &&
            s->value->type & forStatementType){
        addInnerForsBetweenBarriers(*(s->value),
                                    s->value->statementStart,
                                    innerDim);

        s = s->right;
      }

      if(s == NULL)
        return;

      for(int i = innerDim; 0 <= i; --i){
        statement *newStatement = new statement(s->value->depth,
                                                occaForType, &origin,
                                                NULL, NULL);

        newStatement->nodeStart = new strNode("occaInnerFor");
        newStatement->nodeStart->value += '0' + i;
        newStatement->nodeStart->type   = occaForType;

        newStatement->nodeEnd = newStatement->nodeStart;

        if(i == innerDim){
          outerMostLoop = newStatement;
          innerMostLoop = outerMostLoop;
        }
        else{
          innerMostLoop->addStatement(newStatement);
          innerMostLoop->statementStart->value->up = innerMostLoop;
          innerMostLoop = innerMostLoop->statementStart->value;
        }
      }

      statementNode *includeStart = s;

      // Keep privates and shared outside inner loops
      while(includeStart->value->hasDescriptorVariable("occaShared") ||
            includeStart->value->hasDescriptorVariable("exclusive")){

        includeStart = includeStart->right;
      }

      statementNode *includeEnd = includeStart;
      statementNode *stoppedAtNode = NULL;

      while(includeEnd                                   &&
            !statementHasBarrier( *(includeEnd->value) ) &&
            !(includeEnd->value->type & forStatementType)){

        includeEnd = includeEnd->right;
      }

      if(includeEnd){
        stoppedAtNode = includeEnd;
        includeEnd = includeEnd->left;
      }

      // Put the loop node on the origin's statements
      //   and remove nodes that are going to be put
      //   in the inner loops
      statementNode *outerMostLoopSN = new statementNode(outerMostLoop);

      if(origin.statementStart == includeStart)
        origin.statementStart = outerMostLoopSN;

      outerMostLoopSN->right = stoppedAtNode;

      if(includeStart &&
         includeStart->left)
        includeStart->left->right = outerMostLoopSN;

      if(stoppedAtNode)
        stoppedAtNode->left = outerMostLoopSN;

      if(includeStart)
        includeStart->left = NULL;

      if(includeEnd)
        includeEnd->right = NULL;

      innerMostLoop->statementStart = includeStart;

      while(includeStart != includeEnd){
        includeStart->value->up = innerMostLoop;
        includeStart = includeStart->right;
      }

      if(includeEnd)
        includeStart->value->up = innerMostLoop;

      // Increment the depth of statements in the loops
      for(int i = 0; i < innerDim; ++i){
        outerMostLoop = outerMostLoop->statementStart->value;
        applyToAllStatements(*outerMostLoop, &parserBase::incrementDepth);
      }

      applyToAllStatements(*outerMostLoop, &parserBase::incrementDepth);
      --(outerMostLoop->depth);

      // Stick loops on the next part
      if(stoppedAtNode)
        addInnerForsBetweenBarriers(origin, stoppedAtNode, innerDim);
    }

    inline void parserBase::addInnerFors(statement &s){
      int innerDim = getKernelInnerDim(s);

      if(innerDim == -1){
        std::cout << "OCCA Inner for-loop count could not be calculated\n";
        throw 1;
      }

      // Get path and ignore kernel
      statementNode *sPath = findStatementWith(s, &parserBase::statementHasBarrier);

      checkPathForConditionals(sPath);

      loopSection_t loopSection;
      findLoopSections(s, sPath, loopSection);

      // Get private and shared vars
      varInfoNode *varRoot = findVarsMovingToTop(s, loopSection);
      varInfoNode *varPos  = varRoot;

      statementNode *newStatementStart = NULL;
      statementNode *newStatementPos   = NULL;

      while(varPos){
        statement *origin = (varOriginMap[varPos->value]);

        if(origin->type & functionStatementType){
          varPos = varPos->right;
          continue;
        }

        varInfo &info     = *(varPos->value);

        bool initWithValue = false;
        bool infoHasShared = info.hasDescriptor("occaShared");

        if(!infoHasShared){
          strNode *nodePos = origin->nodeStart;
          int declPos = 0;

          while(nodePos){
            if(nodePos->type & unknownVariable){
              if(nodePos->value == info.name)
                break;

              ++declPos;
            }

            nodePos = nodePos->right;
          }

          if((nodePos->right) &&
             (nodePos->right->value == "="))
            initWithValue = true;

          splitDefineForVariable(origin,
                                 info, nodePos,
                                 declPos);
        }
        else{
          statement &originUp  = *(origin->up);
          statementNode *snPos = NULL;

          snPos = originUp.statementStart;

          while(snPos->value != origin)
            snPos = snPos->right;

          if(snPos == originUp.statementStart)
            originUp.statementStart = originUp.statementStart->right;

          snPos->pop();
        }

        varPos = varPos->right;
      }

      addInnerForsBetweenBarriers(s, s.statementStart, innerDim);
    }

    inline void parserBase::addOuterFors(statement &s){
      int outerDim = getKernelOuterDim(s);

      if(outerDim == -1){
        std::cout << "OCCA Outer for-loop count could not be calculated\n";
        throw 1;
      }

      const int occaForType = keywordType["occaOuterFor0"];

      statement *sPos = &s;

      for(int o = outerDim; 0 <= o; --o){
        statement *newStatement = new statement(sPos->depth + 1,
                                                occaForType, &s,
                                                NULL, NULL);

        newStatement->nodeStart = new strNode("occaOuterFor");
        newStatement->nodeStart->value += '0' + o;
        newStatement->nodeStart->type   = occaForType;

        newStatement->nodeEnd = newStatement->nodeStart;

        newStatement->scopeVarMap = sPos->scopeVarMap;

        statementNode *sn = sPos->statementStart;

        while(sn){
          newStatement->addStatement(sn->value);

          sn->value->up = newStatement;
          applyToAllStatements(*(sn->value), &parserBase::incrementDepth);

          statementNode *sn2 = sn->right;
          delete sn;
          sn = sn2;
        }

        sPos->statementCount = 0;
        sPos->statementStart = sPos->statementEnd = NULL;
        sPos->scopeVarMap.clear();

        sPos->addStatement(newStatement);

        sPos = newStatement;
      }
    }

    inline void parserBase::addOccaForsToKernel(statement &s){
      if(s.statementStart == NULL)
        return;

      if(statementKernelUsesNativeOCCA(s))
        return;

      statement *sPos = &s;

      // Get rid of empty blocks
      //  kernel void blah(){{  -->  kernel void blah(){
      //  }}                    -->  }
      while(sPos->statementCount == 1){
        statement *sDown = sPos->statementStart->value;

        if(sDown->type == blockStatementType){
          sPos->scopeVarMap.insert(sDown->scopeVarMap.begin(),
                                   sDown->scopeVarMap.end());

          sPos->statementCount = 0;
          sPos->statementStart = sPos->statementEnd = NULL;

          statementNode *sn = sDown->statementStart;

          while(sn){
            sPos->addStatement(sn->value);

            sn->value->up = sPos;
            applyToAllStatements(*(sn->value), &parserBase::decrementDepth);

            statementNode *sn2 = sn->right;
            delete sn;
            sn = sn2;
          }
        }
        else
          break;
      }

      addInnerFors(s);
      addOuterFors(s);
    }

    inline void parserBase::addOccaFors(statement &globalScope){
      statementNode *statementPos = globalScope.statementStart;

      while(statementPos){
        statement *s = statementPos->value;

        if(statementIsAKernel(*s) &&
           !statementKernelUsesNativeOKL(*s)){

          addOccaForsToKernel(*s);
        }

        statementPos = statementPos->right;
      }
    }

    inline void parserBase::setupOccaVariables(statement &s){
      const int idKeywordType = keywordType["occaInnerId0"];

      strNode *nodePos = s.nodeStart;

      while(nodePos){
        if(nodePos->type & idKeywordType){
          bool isInnerId = ((nodePos->value == "occaInnerId0") ||
                            (nodePos->value == "occaInnerId1") ||
                            (nodePos->value == "occaInnerId2"));

          bool isOuterId = ((nodePos->value == "occaOuterId0") ||
                            (nodePos->value == "occaOuterId1") ||
                            (nodePos->value == "occaOuterId2"));

          bool isInnerDim = ((nodePos->value == "occaInnerDim0") ||
                             (nodePos->value == "occaInnerDim1") ||
                             (nodePos->value == "occaInnerDim2"));

          bool isOuterDim = ((nodePos->value == "occaOuterDim0") ||
                             (nodePos->value == "occaOuterDim1") ||
                             (nodePos->value == "occaOuterDim2"));

          if(isInnerId  || isOuterId ||
             isInnerDim || isOuterDim){
            std::string ioLoop, loopNest;

            if(isInnerId || isOuterId){
              // [occa][-----][Id#]
              ioLoop = nodePos->value.substr(4,5);
              // [occa][-----Id][#]
              loopNest = nodePos->value.substr(11,1);
            }
            else{
              // [occa][-----][Dim#]
              ioLoop = nodePos->value.substr(4,5);
              // [occa][-----Dim][#]
              loopNest = nodePos->value.substr(12,1);
            }

            addOccaForCounter(s, ioLoop, loopNest);
          }
        }

        nodePos = nodePos->right;
      }
    }
    //==============================================


    //---[ Parser Functions ]-----------------------
    void initKeywords();
    void initMacros();

    strNode* splitFileContents(const char *cRoot){
      const char *c = cRoot;

      strNode *nodeRoot = new strNode();
      strNode *nodePos  = nodeRoot;

      int status = readingCode;

      while(*c != '\0'){
        const char *cEnd = readLine(c);

        std::string line = strip(c, cEnd - c);

        if(line.size()){
          if(status != insideCommentBlock){
            status = stripComments(line);
            strip(line);

            if(line.size())
              nodePos = nodePos->push(line);
          }
          else{
            status = stripComments(line);
            strip(line);

            if((status == finishedCommentBlock) && line.size())
              nodePos = nodePos->push(line);
          }
        }

        c = cEnd;
      }

      popAndGoRight(nodeRoot);

      return nodeRoot;
    }

    strNode* labelCode(strNode *lineNodeRoot){
      strNode *nodeRoot = new strNode();
      strNode *nodePos  = nodeRoot;

      strNode *lineNodePos = lineNodeRoot;

      int depth = 0;

      while(lineNodePos){
        const std::string &line = lineNodePos->value;
        const char *cLeft = line.c_str();

        while(*cLeft != '\0'){
          skipWhitespace(cLeft);

          const char *cRight = cLeft;

          if(isAString(cLeft)){
            skipString(cRight);

            nodePos->push( std::string(cLeft, (cRight - cLeft)) );

            nodePos = nodePos->right;

            nodePos->type  = presetValue;
            nodePos->depth = depth;

            cLeft = cRight;
          }
          else if(isANumber(cLeft)){
            skipNumber(cRight);

            nodePos->push( std::string(cLeft, (cRight - cLeft)) );

            nodePos = nodePos->right;

            nodePos->type  = presetValue;
            nodePos->depth = depth;

            cLeft = cRight;
          }
          else{
            const int delimeterChars = isAWordDelimeter(cLeft);

            if(delimeterChars){
              strNode *newNode = new strNode(std::string(cLeft, delimeterChars));

              newNode->type = keywordType[newNode->value];

              newNode->depth = depth;

              if(newNode->type & startSection){
                ++depth;

                nodePos = nodePos->pushDown(newNode);
              }
              else if(newNode->type & endSection){
                nodePos = nodePos->push(newNode);

                --depth;
                nodePos = nodePos->up;
              }
              else
                nodePos = nodePos->push(newNode);

              cLeft += delimeterChars;
            }
            else{
              skipWord(cRight);

              nodePos->push( std::string(cLeft, (cRight - cLeft)) );

              nodePos = nodePos->right;

              keywordTypeMapIterator it = keywordType.find(nodePos->value);

              if(it == keywordType.end())
                nodePos->type = unknownVariable;
              else{
                nodePos->type = it->second;

                // Merge [else] [if] -> [else if]
                if((nodePos->type & flowControlType)       &&
                   (nodePos->left)                         &&
                   (nodePos->left->type & flowControlType) &&
                   ((nodePos->value == "if")         &&
                    (nodePos->left->value == "else") &&
                    (nodePos->left->down.size() == 0))){

                  nodePos->value = "else if";

                  strNode *elseNode = nodePos->left;

                  nodePos->left        = nodePos->left->left;
                  nodePos->left->right = nodePos;

                  delete elseNode->pop();
                }
              }

              nodePos->depth = depth;

              cLeft = cRight;
            }
          }
        }

        lineNodePos = lineNodePos->right;
      }

      if((nodePos != nodeRoot) &&
         (nodeRoot->down.size() == 0))
        popAndGoRight(nodeRoot);

      free(lineNodeRoot);

      return nodeRoot;
    }

    varInfo loadVarInfo(strNode *&nodePos){
      varInfo info;

      while(nodePos &&
            !(nodePos->type & (presetValue | unknownVariable))){

        if(nodePos->type & qualifierType){

          if(nodePos->value == "*"){
            info.typeInfo |= heapPointerType;
            ++info.pointerCount;

            if(nodePos->right &&
               nodePos->right->value == "const"){
              info.typeInfo |= constPointerType;
              nodePos = nodePos->right;
            }
          }
          else if(nodePos->value == "&")
            info.typeInfo |= referenceType;
          else{
            if((nodePos->value  == "texture")              ||
               ((nodePos->value == "image1d_t")            ||
                (nodePos->value == "image2d_t"))           ||
               (nodePos->value  == "cudaSurfaceObject_t"))
              info.typeInfo |= textureType;

            info.descriptors.push_back(*nodePos);
          }
        }
        else if(nodePos->type & specifierType)
          info.type = *nodePos;

        nodePos = nodePos->right;
      }

      if(nodePos == NULL)
        return info;

      info.name = *nodePos;

      const int downCount = nodePos->down.size();

      if(downCount){
        if(nodePos->down[0]->type == startParentheses){
          strNode *argPos = nodePos->down[0];
          info.typeInfo |= functionType;

          strNode *lastPos = lastNode(nodePos);

          // Distinguish between prototypes and function calls
          if(lastPos->value == ";"){
            if(info.type.size())
              info.typeInfo |= protoType;
            else
              info.typeInfo |= functionCallType;
          }
        }
        else if(nodePos->down[0]->type == startBracket){
          info.typeInfo |= (variableType | stackPointerType);

          for(int i = 0; i < downCount; ++i){
            nodePos->down[i]->type == startBracket;

            std::string sps = prettyString(nodePos->down[i]);
            sps = sps.substr(1, sps.size() - 2); // Remove '[' and ']'

            info.stackPointerSizes.push_back(sps);
          }
        }
      }
      else{
        info.typeInfo |= variableType;
        nodePos = nodePos->right;
      }

      return info;
    }

    int statementType(strNode *&nodeRoot){
      if(nodeRoot == NULL){
        std::cout << "Not a valid statement\n";
        throw 1;
      }

      if(nodeRoot->type == keywordType["occaOuterFor0"])
        return keywordType["occaOuterFor0"];

      if(nodeRoot->type & structType)
        return structStatementType;

      else if(nodeRoot->type & unknownVariable){
        if(nodeRoot->right &&
           nodeRoot->right->value == ":"){
          nodeRoot = nodeRoot->right;
          return gotoStatementType;
        }

        while(nodeRoot){
          if(nodeRoot->type & endStatement)
            break;

          nodeRoot = nodeRoot->right;
        }

        return updateStatementType;
      }

      else if(nodeRoot->type & descriptorType){

        while(nodeRoot){
          if(nodeRoot->type & endStatement)
            return declareStatementType;

          // Case:
          //   const varName = ();
          if(nodeRoot->type & operatorType){
            while(nodeRoot){
              if(nodeRoot->type & endStatement)
                break;

              nodeRoot = nodeRoot->right;
            }

            return declareStatementType;
          }

          else if(nodeRoot->down.size() &&
                  (nodeRoot->down[0]->type & parentheses)){

            const int downCount = nodeRoot->down.size();

            if(downCount == 1){
              while(nodeRoot){
                if(nodeRoot->type & endStatement)
                  break;

                if(nodeRoot->right == NULL){
                  std::cout << "Missing a [;] after:\n"
                            << prettyString(nodeRoot, "  ");
                  throw 1;
                }

                nodeRoot = nodeRoot->right;
              }

              return functionPrototypeType;
            }
            else
              return functionDefinitionType;
          }

          if(nodeRoot->right == NULL)
            break;

          nodeRoot = nodeRoot->right;
        }

        return declareStatementType;
      }
      else if(nodeRoot->type & flowControlType){
        if(nodeRoot->value == "for")
          return forStatementType;
        else if(nodeRoot->value == "while")
          return whileStatementType;
        else if(nodeRoot->value == "do")
          return doWhileStatementType;
        else if(nodeRoot->value == "if")
          return ifStatementType;
        else if(nodeRoot->value == "else if")
          return elseIfStatementType;
        else if(nodeRoot->value == "else")
          return elseStatementType;
        else if(nodeRoot->value == "switch")
          return switchStatementType;
      }
      else if(nodeRoot->type & occaKeywordType){
        if((nodeRoot->value.find("occaInnerFor") != std::string::npos) &&
           ((nodeRoot->value == "occaInnerFor0") ||
            (nodeRoot->value == "occaInnerFor1") ||
            (nodeRoot->value == "occaInnerFor2")))

          return keywordType["occaInnerFor0"];

        else if((nodeRoot->value.find("occaOuterFor") != std::string::npos) &&
                ((nodeRoot->value == "occaOuterFor0") ||
                 (nodeRoot->value == "occaOuterFor1") ||
                 (nodeRoot->value == "occaOuterFor2")))

          return keywordType["occaOuterFor0"];
      }
      if(nodeRoot->type & specialKeywordType){
        while(nodeRoot){
          if(nodeRoot->type & endStatement)
            break;

          nodeRoot = nodeRoot->right;
        }

        return blankStatementType;
      }
      if((nodeRoot->type == startBrace) &&
         (nodeRoot->up)                 &&
         !(nodeRoot->up->type & operatorType)){

        nodeRoot = lastNode(nodeRoot);

        return blockStatementType;
      }

      while(nodeRoot &&
            !(nodeRoot->type & endStatement))
        nodeRoot = nodeRoot->right;

      return declareStatementType;
    }

    void initKeywords(){
      keywordsAreInitialized = true;

      //---[ Operator Info ]--------------
      keywordType["!"]  = lUnitaryOperatorType;
      keywordType["%"]  = binaryOperatorType;
      keywordType["&"]  = (binaryOperatorType | qualifierType);
      keywordType["("]  = startParentheses;
      keywordType[")"]  = endParentheses;
      keywordType["*"]  = (binaryOperatorType | qualifierType);
      keywordType["+"]  = (lUnitaryOperatorType | binaryOperatorType);
      keywordType[","]  = binaryOperatorType;
      keywordType["-"]  = (lUnitaryOperatorType | binaryOperatorType);
      keywordType["."]  = binaryOperatorType;
      keywordType["/"]  = binaryOperatorType;
      keywordType[":"]  = endStatement;
      keywordType[";"]  = endStatement;
      keywordType["<"]  = binaryOperatorType;
      keywordType["="]  = binaryOperatorType;
      keywordType[">"]  = binaryOperatorType;
      keywordType["?"]  = ternaryOperatorType;
      keywordType["["]  = startBracket;
      keywordType["]"]  = endBracket;
      keywordType["^"]  = binaryOperatorType;
      keywordType["{"]  = startBrace;
      keywordType["|"]  = binaryOperatorType;
      keywordType["}"]  = endBrace;
      keywordType["~"]  = lUnitaryOperatorType;
      keywordType["!="] = assOperatorType;
      keywordType["%="] = assOperatorType;
      keywordType["&&"] = binaryOperatorType;
      keywordType["&="] = assOperatorType;
      keywordType["*="] = assOperatorType;
      keywordType["+="] = assOperatorType;
      keywordType["++"] = unitaryOperatorType;
      keywordType["-="] = assOperatorType;
      keywordType["--"] = unitaryOperatorType;
      keywordType["->"] = binaryOperatorType;
      keywordType["/="] = assOperatorType;
      keywordType["::"] = binaryOperatorType;
      keywordType["<<"] = binaryOperatorType;
      keywordType["<="] = binaryOperatorType;
      keywordType["=="] = binaryOperatorType;
      keywordType[">="] = binaryOperatorType;
      keywordType[">>"] = binaryOperatorType;
      keywordType["^="] = assOperatorType;
      keywordType["|="] = assOperatorType;
      keywordType["||"] = binaryOperatorType;

      //---[ Types & Specifiers ]---------
      std::string suffix[7] = {"", "1", "2", "3", "4", "8", "16"};

      for(int i = 0; i < 7; ++i){
        keywordType[std::string("int")    + suffix[i]] = specifierType;
        keywordType[std::string("bool")   + suffix[i]] = specifierType;
        keywordType[std::string("char")   + suffix[i]] = specifierType;
        keywordType[std::string("long")   + suffix[i]] = specifierType;
        keywordType[std::string("short")  + suffix[i]] = specifierType;
        keywordType[std::string("float")  + suffix[i]] = specifierType;
        keywordType[std::string("double") + suffix[i]] = specifierType;
      }

      keywordType["void"] = specifierType;

      keywordType["signed"]   = qualifierType;
      keywordType["unsigned"] = qualifierType;

      keywordType["inline"] = qualifierType;
      keywordType["static"] = qualifierType;

      keywordType["const"]    = (qualifierType | occaKeywordType);
      keywordType["restrict"] = (qualifierType | occaKeywordType);
      keywordType["volatile"] = (qualifierType | occaKeywordType);
      keywordType["aligned"]  = (qualifierType | occaKeywordType);

      keywordType["occaConst"]    = (qualifierType | occaKeywordType);
      keywordType["occaRestrict"] = (qualifierType | occaKeywordType);
      keywordType["occaVolatile"] = (qualifierType | occaKeywordType);
      keywordType["occaAligned"]  = (qualifierType | occaKeywordType);
      keywordType["occaConstant"] = (qualifierType | occaKeywordType);

      keywordType["enum"]    = (specifierType | structType);
      keywordType["class"]   = (specifierType | structType);
      keywordType["union"]   = (specifierType | structType);
      keywordType["struct"]  = (specifierType | structType);
      keywordType["typedef"] = (specifierType | structType);
      keywordType["extern"]  = (specifierType | structType);

      //---[ Constants ]------------------
      keywordType["true"]  = presetValue;
      keywordType["false"] = presetValue;

      //---[ Flow Control ]---------------
      keywordType["if"]   = flowControlType;
      keywordType["else"] = flowControlType;

      keywordType["for"] = flowControlType;

      keywordType["do"]    = flowControlType;
      keywordType["while"] = flowControlType;

      keywordType["switch"]  = flowControlType;
      keywordType["case"]    = specialKeywordType;
      keywordType["default"] = specialKeywordType;

      keywordType["break"]    = specialKeywordType;
      keywordType["continue"] = specialKeywordType;
      keywordType["return"]   = specialKeywordType;
      keywordType["goto"]     = specialKeywordType;

      // barrier: auto close if it's not properly done
      //          or give an error

      // const -> constant if it's in global space
      // keywordType["constant"] = occaKeywordType;

      // occaDeviceFunction if not specified as a kernel
      // Add prototype before it

      // Kernel:
      //   Arg is     a pointer -> occaPointer  before type
      //   Arg is not a pointer -> occavariable before name

      //---[ OCCA Keywords ]--------------
      keywordType["kernel"]    = (qualifierType | occaKeywordType);
      keywordType["texture"]   = (qualifierType | occaKeywordType);
      keywordType["shared"]    = (qualifierType | occaKeywordType);
      keywordType["exclusive"] = (qualifierType | occaKeywordType);

      keywordType["occaKernel"]   = (qualifierType | occaKeywordType);
      keywordType["occaFunction"] = (qualifierType | occaKeywordType);
      keywordType["occaPointer"]  = (qualifierType | occaKeywordType);
      keywordType["occaVariable"] = (qualifierType | occaKeywordType);
      keywordType["occaShared"]   = (qualifierType | occaKeywordType);

      keywordType["occaKernelInfoArg"] = (presetValue | occaKeywordType);
      keywordType["occaKernelInfo"]    = (presetValue | occaKeywordType);

      keywordType["barrier"]        = (occaKeywordType | presetValue);
      keywordType["localMemFence"]  = (occaKeywordType | presetValue);
      keywordType["globalMemFence"] = (occaKeywordType | presetValue);

      keywordType["occaBarrier"]        = (occaKeywordType | presetValue);
      keywordType["occaLocalMemFence"]  = (occaKeywordType | presetValue);
      keywordType["occaGlobalMemFence"] = (occaKeywordType | presetValue);

      keywordType["occaInnerFor0"] = (occaStatementType | forStatementType);
      keywordType["occaInnerFor1"] = (occaStatementType | forStatementType);
      keywordType["occaInnerFor2"] = (occaStatementType | forStatementType);

      keywordType["occaOuterFor0"] = (occaStatementType | forStatementType);
      keywordType["occaOuterFor1"] = (occaStatementType | forStatementType);
      keywordType["occaOuterFor2"] = (occaStatementType | forStatementType);

      keywordType["occaInnerId0"] = (occaKeywordType | presetValue);
      keywordType["occaInnerId1"] = (occaKeywordType | presetValue);
      keywordType["occaInnerId2"] = (occaKeywordType | presetValue);

      keywordType["occaOuterId0"] = (occaKeywordType | presetValue);
      keywordType["occaOuterId1"] = (occaKeywordType | presetValue);
      keywordType["occaOuterId2"] = (occaKeywordType | presetValue);

      keywordType["occaGlobalId0"] = (occaKeywordType | presetValue);
      keywordType["occaGlobalId1"] = (occaKeywordType | presetValue);
      keywordType["occaGlobalId2"] = (occaKeywordType | presetValue);

      keywordType["occaInnerDim0"] = (occaKeywordType | presetValue);
      keywordType["occaInnerDim1"] = (occaKeywordType | presetValue);
      keywordType["occaInnerDim2"] = (occaKeywordType | presetValue);

      keywordType["occaOuterDim0"] = (occaKeywordType | presetValue);
      keywordType["occaOuterDim1"] = (occaKeywordType | presetValue);
      keywordType["occaOuterDim2"] = (occaKeywordType | presetValue);

      keywordType["occaGlobalDim0"] = (occaKeywordType | presetValue);
      keywordType["occaGlobalDim1"] = (occaKeywordType | presetValue);
      keywordType["occaGlobalDim2"] = (occaKeywordType | presetValue);

      //---[ CUDA Keywords ]--------------
      keywordType["threadIdx"] = (cudaKeywordType | presetValue);
      keywordType["blockDim"]  = (cudaKeywordType | presetValue);
      keywordType["blockIdx"]  = (cudaKeywordType | presetValue);
      keywordType["gridDim"]   = (cudaKeywordType | presetValue);

      std::string mathFunctions[16] = {
        "sqrt", "sin"  , "asin" ,
        "sinh", "asinh", "cos"  ,
        "acos", "cosh" , "acosh",
        "tan" , "atan" , "tanh" ,
        "atanh", "exp" , "log2" ,
        "log10"
      };

      for(int i = 0; i < 16; ++i){
        std::string mf = mathFunctions[i];
        std::string cmf = mf;
        cmf[0] += ('A' - 'a');

        keywordType["occa"       + cmf] = presetValue;
        keywordType["occaFast"   + cmf] = presetValue;
        keywordType["occaNative" + cmf] = presetValue;
      }

      //---[ Operator Precedence ]--------
      opPrecedence[opHolder("::", binaryOperatorType)]   = 0;

      opPrecedence[opHolder("++", rUnitaryOperatorType)] = 1;
      opPrecedence[opHolder("--", rUnitaryOperatorType)] = 1;
      opPrecedence[opHolder("." , binaryOperatorType)]   = 1;
      opPrecedence[opHolder("->", binaryOperatorType)]   = 1;

      opPrecedence[opHolder("++", lUnitaryOperatorType)] = 2;
      opPrecedence[opHolder("--", lUnitaryOperatorType)] = 2;
      opPrecedence[opHolder("+" , lUnitaryOperatorType)] = 2;
      opPrecedence[opHolder("-" , lUnitaryOperatorType)] = 2;
      opPrecedence[opHolder("!" , lUnitaryOperatorType)] = 2;
      opPrecedence[opHolder("~" , lUnitaryOperatorType)] = 2;
      opPrecedence[opHolder("*" , qualifierType)]        = 2;
      opPrecedence[opHolder("&" , qualifierType)]        = 2;

      opPrecedence[opHolder("*" , binaryOperatorType)]   = 3;
      opPrecedence[opHolder("/" , binaryOperatorType)]   = 3;
      opPrecedence[opHolder("%" , binaryOperatorType)]   = 3;

      opPrecedence[opHolder("+" , binaryOperatorType)]   = 4;
      opPrecedence[opHolder("-" , binaryOperatorType)]   = 4;

      opPrecedence[opHolder("<<", binaryOperatorType)]   = 5;
      opPrecedence[opHolder(">>", binaryOperatorType)]   = 5;

      opPrecedence[opHolder("<" , binaryOperatorType)]   = 6;
      opPrecedence[opHolder("<=", binaryOperatorType)]   = 6;
      opPrecedence[opHolder(">=", binaryOperatorType)]   = 6;
      opPrecedence[opHolder(">" , binaryOperatorType)]   = 6;

      opPrecedence[opHolder("==", binaryOperatorType)]   = 7;
      opPrecedence[opHolder("!=", binaryOperatorType)]   = 7;

      opPrecedence[opHolder("&" , binaryOperatorType)]   = 8;

      opPrecedence[opHolder("^" , binaryOperatorType)]   = 9;

      opPrecedence[opHolder("|" , binaryOperatorType)]   = 10;

      opPrecedence[opHolder("&&", binaryOperatorType)]   = 11;

      opPrecedence[opHolder("||", binaryOperatorType)]   = 12;

      opPrecedence[opHolder("?" , ternaryOperatorType)]  = 13;
      opPrecedence[opHolder("=" , assOperatorType)]      = 13;

      opPrecedence[opHolder("%=", assOperatorType)]      = 14;
      opPrecedence[opHolder("&=", assOperatorType)]      = 14;
      opPrecedence[opHolder("*=", assOperatorType)]      = 14;
      opPrecedence[opHolder("+=", assOperatorType)]      = 14;
      opPrecedence[opHolder("-=", assOperatorType)]      = 14;
      opPrecedence[opHolder("/=", assOperatorType)]      = 14;
      opPrecedence[opHolder("^=", assOperatorType)]      = 14;
      opPrecedence[opHolder("|=", assOperatorType)]      = 14;

      opPrecedence[opHolder("," , binaryOperatorType)]   = 16;

      /*---[ Future Ones ]----------------
        keywordType["using"] = ;
        keywordType["namespace"] = ;
        keywordType["template"] = ;
        ================================*/
    }
    //==============================================

    const std::string parserBase::parseSource(const char *cRoot){
      if(!keywordsAreInitialized){
        initKeywords();
        initMacros();
      }

      strNode *nodeRoot = splitFileContents(cRoot);

      nodeRoot = preprocessMacros(nodeRoot);
      nodeRoot = labelCode(nodeRoot);
      // nodeRoot->print();
      // throw 1;

      statement globalScope(*this);
      globalScope.loadAllFromNode(nodeRoot);

      applyToAllStatements(globalScope, &parserBase::labelKernelsAsNativeOrNot);

      applyToAllStatements(globalScope, &parserBase::setupCudaVariables);
      applyToAllStatements(globalScope, &parserBase::setupOccaVariables);

      applyToAllStatements(globalScope, &parserBase::setupOccaFors);
      applyToAllStatements(globalScope, &parserBase::loadVariableInformation);

      applyToAllStatements(globalScope, &parserBase::addFunctionPrototypes);
      applyToAllStatements(globalScope, &parserBase::updateConstToConstant);

      addOccaFors(globalScope);

      // Also auto-adds barriers if needed
      applyToAllStatements(globalScope, &parserBase::fixOccaForOrder);
      applyToAllStatements(globalScope, &parserBase::addParallelFors);

      applyToAllStatements(globalScope, &parserBase::modifyExclusiveVariables);

      modifyOccaForVariables();
      modifyTextureVariables();

      applyToStatementsDefiningVar(&parserBase::addArgQualifiers);

      return (std::string) globalScope;
    }

    const std::string parserBase::parseFile(const std::string &filename){
      if(!macrosAreInitialized)
        initMacros();

      const char *cRoot = cReadFile(filename);

      const std::string parsedContent = parseSource(cRoot);

      delete [] cRoot;

      return parsedContent;
    }

    //---[ Macro Parser Functions ]-------
    std::string parserBase::getMacroName(const char *&c){
      const char *cStart = c;
      skipWord(cStart);
      skipWhitespace(cStart);
      c = cStart;
      skipWord(c);

      return std::string(cStart, c - cStart);
    }

    bool parserBase::evaluateMacroStatement(const char *&c){
      skipWhitespace(c);

      if(*c == '\0')
        return false;

      strNode *lineNode = new strNode(c);
      applyMacros(lineNode->value);
      strip(lineNode->value);

      strNode *labelNodeRoot = labelCode(lineNode);
      strNode *labelNodePos  = labelNodeRoot;

      // Check if a variable snuck in
      while(labelNodePos){
        if(labelNodePos->type & unknownVariable){
          std::cout << "Variable [" << *labelNodePos << "] not known at compile time.\n";
          throw 1;
        }

        labelNodePos = labelNodePos->right;
      }

      typeHolder th = evaluateLabelNode(labelNodeRoot);

      return (th.doubleValue() != 0);
    }

    typeHolder parserBase::evaluateLabelNode(strNode *labelNodeRoot){
      if(labelNodeRoot->type & presetValue)
        return typeHolder(*labelNodeRoot);

      strNode *labelNodePos = labelNodeRoot;

      while(labelNodePos){
        if(labelNodePos->down.size()){
          strNode *downNode = labelNodePos->down[0];
          labelNodePos->down.clear();

          if(labelNodePos->type)
            labelNodePos = labelNodePos->push( evaluateLabelNode(downNode) );
          else
            labelNodePos->value = evaluateLabelNode(downNode);

          labelNodePos->type = presetValue;
        }

        if(labelNodePos->right == NULL)
          break;

        labelNodePos = labelNodePos->right;
      }

      strNode *labelNodeEnd = labelNodePos;

      if(labelNodeEnd && labelNodeRoot){
        if((labelNodeRoot->type & startParentheses) &&
           (labelNodeEnd->type  & endParentheses)){

          popAndGoRight(labelNodeRoot);
          labelNodeEnd->pop();
        }
      }

      strNode *minOpNode;
      int minPrecedence, minOpType;

      labelNodePos = labelNodeRoot;

      while(true){
        minOpNode     = NULL;
        minPrecedence = 100;
        minOpType     = -1;

        while(labelNodePos){
          if(labelNodePos->type & operatorType){
            int opType = (labelNodePos->type & operatorType);

            opType &= ~qualifierType;

            if(opType & unitaryOperatorType){
              if((opType & binaryOperatorType) && // + and - operators
                 (labelNodePos->left)          &&
                 (labelNodePos->left->type & presetValue)){

                opType = binaryOperatorType;
              }
              else if((opType & rUnitaryOperatorType) &&
                      (labelNodePos->left)            &&
                      (labelNodePos->left->type & presetValue)){

                opType = rUnitaryOperatorType;
              }
              else if((opType & lUnitaryOperatorType) &&
                      (labelNodePos->right)           &&
                      (labelNodePos->right->type & presetValue)){

                opType = lUnitaryOperatorType;
              }
              else
                opType &= ~unitaryOperatorType;
            }

            const int opP = opPrecedence[opHolder(labelNodePos->value,
                                                  opType)];

            if(opP < minPrecedence){
              minOpType     = opType;
              minOpNode     = labelNodePos;
              minPrecedence = opP;
            }
          }

          labelNodePos = labelNodePos->right;
        }

        if(minOpNode == NULL){
          if(labelNodeRoot && (labelNodeRoot->right == NULL))
            return typeHolder(*labelNodeRoot);

          std::cout << "5. Error on:\n";
          labelNodeRoot->print("  ");
          throw 1;
        }
        else{
          if(minOpType & unitaryOperatorType){
            if(minOpType & lUnitaryOperatorType){
              std::string op = minOpNode->value;
              std::string a  = minOpNode->right->value;

              minOpNode->value = applyOperator(op, a);
              minOpNode->type  = presetValue;

              minOpNode->right->pop();
            }
            else if(minOpType & rUnitaryOperatorType){
              std::cout << "Postfix operator [" << *minOpNode << "] cannot be used in a macro.\n";
              throw 1;
            }
          }
          else if(minOpType & binaryOperatorType){
            minOpNode = minOpNode->left;

            std::string a  = minOpNode->value;
            std::string op = minOpNode->right->value;
            std::string b  = minOpNode->right->right->value;

            minOpNode->value = applyOperator(a, op, b);
            minOpNode->type  = presetValue;

            minOpNode->right->pop();
            minOpNode->right->pop();
          }
          else if(minOpType & ternaryOperatorType){
            minOpNode = minOpNode->left;

            std::string a  = minOpNode->value;
            std::string op = minOpNode->right->value;
            std::string b  = minOpNode->right->right->value;
            std::string c  = minOpNode->right->right->right->right->value;

            minOpNode->value = applyOperator(a, op, b, c);
            minOpNode->type  = presetValue;

            minOpNode->right->pop();
            minOpNode->right->pop();
            minOpNode->right->pop();
            minOpNode->right->pop();
          }
        }

        if(labelNodeRoot->right == NULL)
          return typeHolder(*labelNodeRoot);

        labelNodePos = labelNodeRoot;
      }

      // Shouldn't get here
      typeHolder th(labelNodeRoot->value);

      return th;
    }

    void parserBase::loadMacroInfo(macroInfo &info, const char *&c){
      skipWhitespace(c);

      info.argc = 0;
      info.parts.clear();
      info.argBetweenParts.clear();

      info.parts.push_back(""); // First part

      info.isAFunction = false;

      if(*c == '\0')
        return;

      if(*c != '('){
        const size_t chars = strlen(c);

        info.parts[0] = strip(c, chars);

        c += chars;

        return;
      }

      int partPos = 0;
      info.isAFunction = true;

      ++c; // '('

      typedef std::map<std::string,int> macroArgMap_t;
      typedef macroArgMap_t::iterator macroArgMapIterator;
      macroArgMap_t macroArgMap;

      while(*c != '\0'){
        skipWhitespace(c);
        const char *cStart = c;
        skipWord(c);

        macroArgMap[std::string(cStart, c - cStart)] = (info.argc++);

        skipWhitespace(c);

        if(*(c++) == ')')
          break;
      }

      skipWhitespace(c);

      while(*c != '\0'){
        const char *cStart = c;

        if(isAString(c)){
          skipString(c);

          info.parts[partPos] += std::string(cStart, (c - cStart));
          continue;
        }

        const int delimeterChars = skipWord(c);

        std::string word = std::string(cStart, c - cStart);

        macroArgMapIterator it = macroArgMap.find(word);

        if(it == macroArgMap.end())
          info.parts[partPos] += word;
        else{
          info.argBetweenParts.push_back(it->second);
          info.parts.push_back("");
          ++partPos;
        }

        cStart = c;
        c += delimeterChars;

        if(cStart != c)
          info.parts[partPos] += std::string(cStart, c - cStart);

        skipWhitespace(c);
      }
    }

    int parserBase::loadMacro(const std::string &line, const int state){
      const char *c = (line.c_str() + 1); // 1 = #

      while(*c != '\0'){
        skipWhitespace(c);
        const char *cEnd = c;
        skipToWhitespace(cEnd);

        if(stringsAreEqual(c, (cEnd - c), "if")){
          c = cEnd;

          bool isTrue = evaluateMacroStatement(c);

          if(isTrue)
            return (startHash | readUntilNextHash);
          else
            return (startHash | ignoreUntilNextHash);
        }
        else if(stringsAreEqual(c, (cEnd - c), "elif")){
          if((state & readUntilNextHash) || (state & ignoreUntilEnd))
            return ignoreUntilEnd;

          c = cEnd;

          bool isTrue = evaluateMacroStatement(c);

          if(isTrue)
            return readUntilNextHash;
          else
            return ignoreUntilNextHash;
        }
        else if(stringsAreEqual(c, (cEnd - c), "else")){
          if((state & readUntilNextHash) || (state & ignoreUntilEnd))
            return ignoreUntilEnd;
          else
            return readUntilNextHash;
        }
        else if(stringsAreEqual(c, (cEnd - c), "ifdef")){
          std::string name = getMacroName(c);

          if(macroMap.find(name) != macroMap.end())
            return (startHash | ignoreUntilNextHash);
          else
            return (startHash | readUntilNextHash);
        }
        else if(stringsAreEqual(c, (cEnd - c), "ifndef")){
          std::string name = getMacroName(c);

          if(macroMap.find(name) != macroMap.end())
            return (startHash | readUntilNextHash);
          else
            return (startHash | ignoreUntilNextHash);
        }
        else if(stringsAreEqual(c, (cEnd - c), "endif")){
          return doneIgnoring;
        }
        else if(stringsAreEqual(c, (cEnd - c), "define")){
          if(state & ignoring)
            return state;

          std::string name = getMacroName(c);
          int pos;

          if(macroMap.find(name) == macroMap.end()){
            pos = macros.size();
            macros.push_back( macroInfo() );
            macroMap[name] = pos;
          }
          else
            pos = macroMap[name];

          macroInfo &info = macros[pos];
          info.name = name;

          loadMacroInfo(info, c);
        }
        else if(stringsAreEqual(c, (cEnd - c), "undef")){
          if(state & ignoring)
            return state;

          std::string name = getMacroName(c);

          if(macroMap.find(name) != macroMap.end())
            macroMap.erase(name);
        }
        else if(stringsAreEqual(c, (cEnd - c), "include")){
          if(state & ignoring)
            return state;
        }

        c = cEnd;
      }

      return state;
    }

    void parserBase::applyMacros(std::string &line){
      const char *c = line.c_str();
      std::string newLine = "";

      bool foundMacro = false;

      while(*c != '\0'){
        const char *cStart = c;

        if(isAString(c)){
          skipString(c);

          newLine += std::string(cStart, (c - cStart));
          continue;
        }

        int delimeterChars = skipWord(c);

        std::string word = std::string(cStart, c - cStart);

        macroMapIterator it = macroMap.find(word);

        if((delimeterChars == 2) &&
           stringsAreEqual(c, delimeterChars, "##") &&
           it != macroMap.end()){
          macroInfo &info = macros[it->second];
          word = info.parts[0];
        }

        while((delimeterChars == 2) &&
              stringsAreEqual(c, delimeterChars, "##")){
          c += 2;

          cStart = c;
          delimeterChars = skipWord(c);

          std::string word2 = std::string(cStart, c - cStart);

          it = macroMap.find(word2);

          if(it != macroMap.end()){
            macroInfo &info = macros[it->second];
            word += info.parts[0];
          }
          else
            word += word2;
        }

        it = macroMap.find(word);

        if(it != macroMap.end()){
          foundMacro = true;

          macroInfo &info = macros[it->second];

          if(!info.isAFunction)
            newLine += info.parts[0];

          else{
            std::vector<std::string> args;

            ++c; // '('

            while(*c != '\0'){
              skipWhitespace(c);
              cStart = c;
              skipWord(c);

              args.push_back( std::string(cStart, c - cStart) );

              skipWhitespace(c);

              if(*(c++) == ')')
                break;
            }

            newLine += info.applyArgs(args);
          }
        }
        else
          newLine += word;

        cStart = c;
        c += delimeterChars;

        if(cStart != c)
          newLine += std::string(cStart, c - cStart);

        if(isWhitespace(*c)){
          newLine += ' ';
          skipWhitespace(c);
        }
      }

      line = newLine;

      if(foundMacro)
        applyMacros(line);
    }

    strNode* parserBase::preprocessMacros(strNode *nodeRoot){
      strNode *nodePos  = nodeRoot;

      std::stack<int> statusStack;

      int currentState = doNothing;

      while(nodePos){
        std::string &line = nodePos->value;
        bool ignoreLine = false;

        if(line[0] == '#'){
          const int oldState = currentState;

          currentState = loadMacro(line, currentState);

          // Nested #if's
          if(currentState & startHash){
            currentState &= ~startHash;
            statusStack.push(oldState);
          }

          if(currentState & doneIgnoring){
            statusStack.pop();

            if(statusStack.size())
              currentState = statusStack.top();
            else
              currentState = doNothing;
          }

          ignoreLine = true;
        }
        else{
          if(!(currentState & ignoring))
            applyMacros(line);
          else
            ignoreLine = true;
        }

        if(ignoreLine){
          if(nodeRoot == nodePos)
            nodeRoot = nodePos->right;

          popAndGoRight(nodePos);
        }
        else
          nodePos = nodePos->right;
      }

      return nodeRoot;
    }
    //====================================

    void parserBase::initMacros(){
      //---[ Macros ]---------------------
      loadMacro("#define kernel occaKernel");

      loadMacro("#define barrier        occaBarrier");
      loadMacro("#define localMemFence  occaLocalMemFence");
      loadMacro("#define globalMemFence occaGlobalMemFence");

      loadMacro("#define shared   occaShared");
      loadMacro("#define restrict occaRestrict");
      loadMacro("#define volatile occaVolatile");
      loadMacro("#define aligned  occaAligned");
      loadMacro("#define const    occaConst");
      loadMacro("#define constant occaConstant");

      std::string mathFunctions[16] = {
        "sqrt", "sin"  , "asin" ,
        "sinh", "asinh", "cos"  ,
        "acos", "cosh" , "acosh",
        "tan" , "atan" , "tanh" ,
        "atanh", "exp" , "log2" ,
        "log10"
      };

      for(int i = 0; i < 16; ++i){
        std::string mf = mathFunctions[i];
        std::string cmf = mf;
        cmf[0] += ('A' - 'a');

        loadMacro("#define "       + mf  + " occa"       + cmf);
        loadMacro("#define fast"   + cmf + " occaFast"   + cmf);
        loadMacro("#define native" + cmf + " occaNative" + cmf);
      }

      //---[ CUDA Macros ]----------------
      loadMacro("#define __global__ occaKernel");

      loadMacro("#define __syncthreads()       occaBarrier(occaGlobalMemFence)");
      loadMacro("#define __threadfence_block() occaBarrier(occaLocalMemFence)");
      loadMacro("#define __threadfence()       occaBarrier(occaGlobalMemFence)");

      loadMacro("#define __shared__   occaShared");
      loadMacro("#define __restrict__ occaRestrict");
      loadMacro("#define __volatile__ occaVolatile");
      loadMacro("#define __constant__ occaConstant");

      loadMacro("#define __device__ occaFunction");

      //---[ OpenCL Macros ]--------------
      loadMacro("#define __kernel occaKernel");

      loadMacro("#define CLK_LOCAL_MEM_FENCE  occaLocalMemFence");
      loadMacro("#define CLK_GLOBAL_MEM_FENCE occaGlobalMemFence");

      loadMacro("#define __local    occaShared");
      loadMacro("#define __global   occaPointer");
      loadMacro("#define __constant occaConstant");

      loadMacro("#define get_num_groups(X)  occaOuterDim##X");
      loadMacro("#define get_group_id(X)    occaOuterId##X ");
      loadMacro("#define get_local_size(X)  occaInnerDim##X");
      loadMacro("#define get_local_id(X)    occaInnerId##X ");
      loadMacro("#define get_global_size(X) occaGlobalDim##X");
      loadMacro("#define get_global_id(X)   occaGlobalId##X ");
    }
  };

  // Just to ignore the namespace
  class parser : public parserNamespace::parserBase {};
};

int main(int argc, char **argv){
  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("test.cpp");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("openclTest.cpp");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   occa::parser parser;
  //   std::string parsedContent = parser.parseFile("cudaTest.cpp");
  //   std::cout << parsedContent << '\n';
  // }

  {
    occa::parser parser;
    std::string parsedContent = parser.parseFile("addVectors.okl");
    std::cout << parsedContent << '\n';
  }
}

#endif
