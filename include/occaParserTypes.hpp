#ifndef OCCA_PARSER_TYPES_HEADER
#define OCCA_PARSER_TYPES_HEADER

#include "occaParserDefines.hpp"
#include "occaParserNodes.hpp"
#include "occaParserStatement.hpp"

namespace occa {
  class parsedKernelInfo;

  namespace parserNS {
    class expNode;
    class typeInfo;
    class varInfo;

    class varLeaf_t;

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
                   int leafPos);

      int loadFromFortran(varInfo &var,
                          expNode &expRoot,
                          int leafPos);

      strNode* loadFrom(statement &s,
                        strNode *nodePos);

      strNode* loadFromFortran(varInfo &var,
                               statement &s,
                               strNode *nodePos);

      bool updateFortranVar(varInfo &var,
                            const std::string &fortranQualifier);

      int updateFortranVar(varInfo &var,
                           expNode &expPos,
                           const int leafPos);

      strNode* updateFortranVar(varInfo &var,
                                statement &s,
                                strNode *nodePos);

      //---[ Qualifier Info ]-----------
      bool has(const std::string &qName) const;
      const std::string& get(const int pos) const;

      void add(const std::string &qName,
               int pos = -1);

      void remove(const std::string &qName);
      void remove(const int pos, const int count = 1);

      void clear();
      //================================

      std::string toString();
      operator std::string ();

      friend std::ostream& operator << (std::ostream &out, qualifierInfo &type);
    };
    //============================================


    //---[ Type Info Class ]----------------------
    class typeInfo {
    public:
      qualifierInfo leftQualifiers;
      std::string name;

      int nestedInfoCount;
      expNode *nestedExps;

      bool typedefHasDefinition;
      typeInfo *typedefing;
      typeInfo *baseType;

      varInfo *typedefVar;

      typeInfo();

      typeInfo(const typeInfo &type);
      typeInfo& operator = (const typeInfo &type);

      typeInfo clone();

      //---[ NEW ]------------
      int loadFrom(expNode &expRoot,
                   int leafPos = 0);

      int loadTypedefFrom(expNode &expRoot,
                          int leafPos = 0);

      static int delimiterCount(expNode &expRoot,
                                const char *delimiter);

      static int nextDelimiter(expNode &expRoot,
                               int leafPos,
                               const char *delimiter);

      static bool statementIsATypeInfo(expNode &expRoot,
                                       int leafPos);
      //======================

      static bool statementIsATypeInfo(statement &s,
                                       strNode *nodePos);

      //---[ Type Info ]----------------
      void addQualifier(const std::string &qName,
                        int pos = -1);
      //================================

      std::string toString(const std::string &tab = "");
      operator std::string ();

      friend std::ostream& operator << (std::ostream &out, typeInfo &type);
    };
    //============================================


    //---[ Variable Info Class ]------------------
    namespace varType {
      static const int var             = (1 << 0);

      static const int functionType    = (7 << 1);
      static const int function        = (3 << 1);
      static const int functionDec     = (1 << 1);
      static const int functionDef     = (1 << 2);
      static const int functionPointer = (1 << 3);
    };

    class varInfo {
    public:
      int info;

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

      int argumentCount;
      varInfo **argumentVarInfos;

      int functionNestCount;
      varInfo *functionNests;

      varInfo();

      varInfo(const varInfo &var);
      varInfo& operator = (const varInfo &var);

      varInfo clone();

      static int variablesInStatement(strNode *nodePos);

      //---[ NEW ]----------------------
      int loadFrom(expNode &expRoot,
                   int leafPos,
                   varInfo *varHasType = NULL);

      int loadTypeFrom(expNode &expRoot,
                       int leafPos,
                       varInfo *varHasType = NULL);

      int loadNameFrom(expNode &expRoot,
                       int leafPos);

      int getVarInfoFrom(expNode &expRoot,
                         int leafPos);

      int getNestCountFrom(expNode &expRoot,
                           int leafPos);

      int loadStackPointersFrom(expNode &expRoot,
                                int leafPos);

      int loadArgsFrom(expNode &expRoot,
                       int leafPos);

      //   ---[ Fortran ]-----
      int loadFromFortran(expNode &expRoot,
                          int leafPos,
                          varInfo *varHasType = NULL);

      int loadTypeFromFortran(expNode &expRoot,
                              int leafPos,
                              varInfo *varHasType = NULL);

      static std::string getFullFortranType(expNode &expRoot,
                                            int &leafPos);

      int loadStackPointersFromFortran(expNode &expRoot,
                                       int leafPos);

      void setupFortranStackExp(expNode &stackExp,
                                expNode &valueExp);
      //   ===================
      //================================

      //---[ OLD ]----------------------
      strNode* loadFrom(statement &s,
                        strNode *nodePos,
                        varInfo *varHasType = NULL);

      strNode* loadTypeFrom(statement &s,
                            strNode *nodePos,
                            varInfo *varHasType);

      strNode* loadNameFrom(statement &s,
                            strNode *nodePos);

      int getVarInfoFrom(statement &s,
                         strNode *nodePos);

      int getNestCountFrom(statement &s,
                           strNode *nodePos);

      strNode* loadStackPointersFrom(statement &s,
                                     strNode *nodePos);

      strNode* loadArgsFrom(statement &s,
                            strNode *nodePos);

      //   ---[ Fortran ]-----
      strNode* loadFromFortran(statement &s,
                               strNode *nodePos,
                               varInfo *varHasType = NULL);

      strNode* loadTypeFromFortran(statement &s,
                                   strNode *nodePos,
                                   varInfo *varHasType);

      static std::string getFullFortranType(strNode *&nodePos);
      //   ===================
      //================================

      //---[ Variable Info ]------------
      int leftQualifierCount() const;
      int rightQualifierCount() const;

      bool hasQualifier(const std::string &qName) const;
      bool hasRightQualifier(const std::string &qName) const;

      void addQualifier(const std::string &qName,
                        int pos = -1);

      void addRightQualifier(const std::string &qName,
                             int pos = -1);

      void removeQualifier(const std::string &qName);
      void removeRightQualifier(const std::string &qName);

      const std::string& getLeftQualifier(const int pos) const;
      const std::string& getRightQualifier(const int pos) const;

      const std::string& getLastLeftQualifier() const;
      const std::string& getLastRightQualifier() const;

      expNode& stackSizeExpNode(const int pos);
      void removeStackPointers();

      varInfo& getArgument(const int pos);
      void setArgument(const int pos, varInfo &var);
      void addArgument(const int pos, varInfo &arg);
      //================================

      bool isConst() const;

      std::string toString(const bool printType = true);

      operator std::string ();

      friend std::ostream& operator << (std::ostream &out, varInfo &var);
    };
    //============================================


    //---[ Var Dependency Graph ]-----------------
    class sDep_t {
    public:
      int sID;
      varInfoVector_t deps;

      sDep_t();

      sDep_t(const sDep_t &sd);
      sDep_t& operator = (const sDep_t &sd);

      varInfo& operator [] (const int pos);

      int size();

      void add(varInfo &var);
      void uniqueAdd(varInfo &var);

      bool has(varInfo &var);
    };

    class varDepGraph {
    public:
      std::vector<sDep_t> sUpdates;

      varDepGraph();

      varDepGraph(varInfo &var,
                  statement &sBound);

      varDepGraph(varInfo &var,
                  statement &sBound,
                  statementIdMap_t &idMap);

      varDepGraph(const varDepGraph &vdg);
      varDepGraph& operator = (const varDepGraph &vdg);

      void free();

      void setup(varInfo &var,
                 statement &sBound);

      void setup(varInfo &var,
                 statement &sBound,
                 statementIdMap_t &idMap);

      bool checkStatementForDependency(varInfo &var,
                                       statement &s,
                                       const int sBoundID,
                                       statementIdMap_t &idMap);

      bool has(const int sID);

      void addDependencyMap(idDepMap_t &depMap);

      void addFullDependencyMap(idDepMap_t &depMap,
                                statementIdMap_t &idMap);
      void addFullDependencyMap(idDepMap_t &depMap,
                                statementIdMap_t &idMap,
                                statementVector_t &sVec);
    };
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
