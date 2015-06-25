#ifndef OCCA_PARSER_TYPES_HEADER
#define OCCA_PARSER_TYPES_HEADER

#include "occaParserDefines.hpp"
#include "occaParserNodes.hpp"

namespace occa {
  class parsedKernelInfo;

  namespace parserNS {
    class expNode;
    class typeInfo;
    class varInfo;

    class varLeaf_t;

    bool isInlinedASM(const std::string &attrName);
    bool isInlinedASM(expNode &expRoot, int leafPos);

    //---[ Attribute Class ]----------------------
    bool isAnAttribute(const std::string &attrName);
    bool isAnAttribute(expNode &expRoot, int leafPos);

    int skipAttribute(expNode &expRoot, int leafPos);

    class attribute_t {
    public:
      std::string name;

      int argCount;
      expNode **args;

      expNode *value;

      attribute_t();
      attribute_t(expNode &e);

      attribute_t(const attribute_t &attr);
      attribute_t& operator = (const attribute_t &attr);

      void load(expNode &e);
      void loadVariable(expNode &e);

      expNode& operator [] (const int pos);

      std::string argStr(const int pos);

      expNode& valueExp();
      std::string valueStr();

      operator std::string();
    };

    std::ostream& operator << (std::ostream &out, attribute_t &attr);

    void setAttributeMap(attributeMap_t &attributeMap,
                         const std::string &attrName);

    int setAttributeMap(attributeMap_t &attributeMap,
                        expNode &expRoot,
                        int leafPos);

    int setAttributeMapR(attributeMap_t &attributeMap,
                         expNode &expRoot,
                         int leafPos);

    void printAttributeMap(attributeMap_t &attributeMap);
    std::string attributeMapToString(attributeMap_t &attributeMap);
    //============================================


    //---[ Qualifier Info Class ]-----------------
    class qualifierInfo {
    public:
      int qualifierCount;
      std::string *qualifiers;

      qualifierInfo();

      qualifierInfo(const qualifierInfo &q);
      qualifierInfo& operator = (const qualifierInfo &q);

      inline std::string& operator [] (const int pos){
        return qualifiers[pos];
      }

      inline const std::string& operator [] (const int pos) const {
        return qualifiers[pos];
      }

      inline int size(){
        return qualifierCount;
      }

      void free();

      qualifierInfo clone();

      int loadFrom(expNode &expRoot,
                   int leafPos = 0);

      int loadFrom(varInfo &var,
                   expNode &expRoot,
                   int leafPos = 0);

      int loadFrom(statement &s,
                   expNode &expRoot,
                   int leafPos = 0);

      int loadFrom(statement &s,
                   varInfo &var,
                   expNode &expRoot,
                   int leafPos = 0);

      int loadFromFortran(statement &s,
                          varInfo &var,
                          expNode &expRoot,
                          int leafPos);

      bool fortranVarNeedsUpdate(varInfo &var,
                                 const std::string &fortranQualifier);

      int updateFortranVar(statement &s,
                           varInfo &var,
                           expNode &expPos,
                           const int leafPos);

      //---[ Qualifier Info ]-----------
      int has(const std::string &qName);
      std::string& get(const int pos);

      void add(const std::string &qName,
               int pos = -1);

      void remove(const std::string &qName);
      void remove(const int pos, const int count = 1);

      void clear();

      bool hasImplicitInt();
      //================================

      std::string toString();
      operator std::string ();

      friend std::ostream& operator << (std::ostream &out, qualifierInfo &type);
    };

    bool expHasQualifier(expNode &allExp, int expPos);
    //============================================


    //---[ Type Info Class ]----------------------
    class typeInfo {
    public:
      qualifierInfo leftQualifiers;

      std::string name;

      info_t thType;

      int nestedInfoCount;
      expNode *nestedExps;

      bool typedefHasDefinition;
      typeInfo *typedefing;
      typeInfo *baseType;

      varInfo *typedefVar;

      opOverloadMaps_t opOverloadMaps;

      typeInfo();

      typeInfo(const typeInfo &type);
      typeInfo& operator = (const typeInfo &type);

      typeInfo clone();

      //---[ Load Info ]----------------
      int loadFrom(expNode &expRoot,
                   int leafPos = 0,
                   bool addTypeToScope = false);

      int loadFrom(statement &s,
                   expNode &expRoot,
                   int leafPos = 0,
                   bool addTypeToScope = false);

      int loadTypedefFrom(statement &s,
                          expNode &expRoot,
                          int leafPos = 0);

      void updateThType();

      static bool statementIsATypeInfo(statement &s,
                                       expNode &expRoot,
                                       int leafPos);

      static int delimiterCount(expNode &expRoot,
                                const char *delimiter);

      static int nextDelimiter(expNode &expRoot,
                               int leafPos,
                               const char *delimiter);
      //======================

      //---[ Type Info ]----------------
      int hasQualifier(const std::string &qName);

      void addQualifier(const std::string &qName,
                        int pos = -1);

      bool hasImplicitInt();

      int pointerDepth();
      //================================

      //---[ Class Info ]---------------
      varInfo* hasOperator(const std::string &op);
      //================================

      std::string toString(const std::string &tab = "");
      operator std::string ();

      friend std::ostream& operator << (std::ostream &out, typeInfo &type);
    };
    //============================================


    //---[ Variable Info Class ]------------------
    namespace varType {
      static const int var             = (1 << 0);
      static const int functionPointer = (3 << 0);

      static const int function        = (3 << 2);
      static const int functionDec     = (1 << 2);
      static const int functionDef     = (1 << 3);

      static const int variadic        = (1 << 4);
      static const int block           = (1 << 5);
    };

    class varInfo {
    public:
      int info;

      attributeMap_t attributeMap;
      qualifierInfo leftQualifiers, rightQualifiers;

      typeInfo *baseType;

      std::string name;

      int pointerCount;

      // stackPointersUsed:
      //   0: Only used for keeping sizes
      //   1: Merge array to a 1D array
      //   X: Same as stackPointerCount (default)
      int stackPointerCount, stackPointersUsed;
      expNode *stackExpRoots;

      int bitfieldSize;

      // @dim()
      attribute_t dimAttr;
      intVector_t idxOrdering;

      bool usesTemplate;
      int tArgCount;
      typeInfo **tArgs;

      int argumentCount;
      varInfo **argumentVarInfos;

      int functionNestCount;
      varInfo *functionNests;

      varInfo();

      varInfo(const varInfo &var);
      varInfo& operator = (const varInfo &var);

      varInfo clone();
      varInfo* clonePtr();

      static int variablesInStatement(expNode &expRoot);

      //---[ Load Info ]----------------
      int loadFrom(expNode &expRoot,
                   int leafPos = 0,
                   varInfo *varHasType = NULL);

      int loadFrom(statement &s,
                   expNode &expRoot,
                   int leafPos = 0,
                   varInfo *varHasType = NULL);

      int loadTypeFrom(statement &s,
                       expNode &expRoot,
                       int leafPos,
                       varInfo *varHasType = NULL);

      int loadNameFrom(statement &s,
                       expNode &expRoot,
                       int leafPos);

      bool nodeHasName(expNode &expRoot,
                       int leafPos);

      int loadNameFromNode(expNode &expRoot,
                           int leafPos);

      int getVarInfoFrom(statement &s,
                         expNode &expRoot,
                         int leafPos);

      int getNestCountFrom(expNode &expRoot,
                           int leafPos);

      int loadStackPointersFrom(expNode &expRoot,
                                int leafPos);

      int loadArgsFrom(statement &s,
                       expNode &expRoot,
                       int leafPos);

      void setupAttributes();

      //   ---[ Fortran ]-----
      int loadFromFortran(expNode &expRoot,
                          int leafPos,
                          varInfo *varHasType = NULL);

      int loadFromFortran(statement &s,
                          expNode &expRoot,
                          int leafPos,
                          varInfo *varHasType = NULL);

      int loadTypeFromFortran(expNode &expRoot,
                              int leafPos,
                              varInfo *varHasType = NULL);

      int loadTypeFromFortran(statement &s,
                              expNode &expRoot,
                              int leafPos,
                              varInfo *varHasType = NULL);

      static std::string getFullFortranType(expNode &expRoot,
                                            int &leafPos);

      int loadStackPointersFromFortran(expNode &expRoot,
                                       int leafPos);

      void setupFortranStackExp(expNode &stackExp,
                                expNode &valueExp);
      //   ===================

      void organizeExpNodes();
      //================================

      //---[ Variable Info ]------------
      attribute_t* hasAttribute(const std::string &attr);

      int leftQualifierCount();
      int rightQualifierCount();

      int hasQualifier(const std::string &qName);
      int hasRightQualifier(const std::string &qName);

      void addQualifier(const std::string &qName,
                        int pos = -1);

      void addRightQualifier(const std::string &qName,
                             int pos = -1);

      void removeQualifier(const std::string &qName);
      void removeRightQualifier(const std::string &qName);

      std::string& getLeftQualifier(const int pos);
      std::string& getRightQualifier(const int pos);

      std::string& getLastLeftQualifier();
      std::string& getLastRightQualifier();

      int pointerDepth();

      expNode& stackSizeExpNode(const int pos);
      void removeStackPointers();

      varInfo& getArgument(const int pos);
      void setArgument(const int pos, varInfo &var);
      void addArgument(const int pos, varInfo &arg);
      //================================

      //---[ Class Info ]---------------
      varInfo* hasOperator(const std::string &op);

      bool canBeCastedTo(varInfo &var);
      bool hasSameTypeAs(varInfo &var);
      //================================

      bool isConst();

      void printDebugInfo();
      std::string toString(const bool printType = true,
                           const std::string &tab = "");

      operator std::string ();

      friend std::ostream& operator << (std::ostream &out, varInfo &var);
    };
    //============================================


    //---[ Overloaded Operator Class ]------------
    class overloadedOp_t {
    public:
      std::vector<varInfo*> functions;

      void add(varInfo &function);

      varInfo* getFromArgs(const int argumentCount,
                           expNode *arguments);

      varInfo* getFromTypes(const int argumentCount,
                            varInfo *argumentTypes);

      varInfo* bestFitFor(const int argumentCount,
                          varInfo *argumentTypes,
                          varInfoVector_t &candidates);
    };
    //============================================


    //---[ Function Info Class ]------------------
    class funcInfo {
    public:
      varInfo *var;

      int argCount;
      expNode *args;

      bool usesTemplate;
      int tArgCount;
      typeInfo *tArgs;
    };
    //============================================


    //---[ Var Dependency Graph ]-----------------
#if 0
    class varDepInfo;
    class smntDepInfo;

    typedef node<varDepInfo*>                     varDepInfoNode;
    typedef node<smntDepInfo*>                    smntDepInfoNode;

    typedef std::map<varInfo*,depInfo*>           varToDepMap;
    typedef std::map<statement*,smntDepInfoNode*> smntToVarDepMap;

    namespace depType {
      static const int none   = 0;
      static const int set    = (1 << 0);
      static const int update = (1 << 1);
    }

    class varDepInfo {
    public:
      int info;

      varInfo *var;
      varDepInfoNode *myNode;

      int startInfo();
      int endInfo();
    };

    class smntDepInfo {
    public:
      varToDepMap v2dMap;

      statement *s;
      smntDepInfoNode *myNode;

      varDepInfo* has(varInfo &var);
      varDepInfo& operator () (varInfo &var);
    };

    class depMap_t {
    public:
      smntToVarDepMap map;

      varDepInfo* has(statement &s, varInfo &var);
      varDepInfo& operator () (statement &s, varInfo &var);
    };
#endif
    //============================================


    //---[ Kernel Info ]--------------------------
    class argumentInfo {
    public:
      int pos;
      bool isConst;

      argumentInfo();

      argumentInfo(const argumentInfo &info);
      argumentInfo& operator = (const argumentInfo &info);
    };

    class kernelInfo {
    public:
      std::string name;
      std::string baseName;

      std::vector<statement*> nestedKernels;
      std::vector<argumentInfo> argumentInfos;

      kernelInfo();

      kernelInfo(const kernelInfo &info);
      kernelInfo& operator = (const kernelInfo &info);

      occa::parsedKernelInfo makeParsedKernelInfo();
    };
    //==============================================
  };

  //---[ Parsed Kernel Info ]---------------------
  typedef parserNS::argumentInfo argumentInfo;

  class parsedKernelInfo {
  public:
    std::string name, baseName;
    int nestedKernels;

    std::vector<argumentInfo> argumentInfos;

    parsedKernelInfo();

    parsedKernelInfo(const parsedKernelInfo & kInfo);
    parsedKernelInfo& operator = (const parsedKernelInfo & kInfo);

    void removeArg(const int pos);

    inline bool argIsConst(const int pos) const {
      if(((size_t) pos) < argumentInfos.size())
        return argumentInfos[pos].isConst;

      return false;
    }
  };
  //==============================================
};

#endif
