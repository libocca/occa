#ifndef OCCA_PARSER_TYPES_HEADER
#define OCCA_PARSER_TYPES_HEADER

#include "occaParserDefines.hpp"
#include "occaParserNodes.hpp"
#include "occaParserStatement.hpp"

namespace occa {
  namespace parserNamespace {
    class expNode;
    class typeInfo;
    class varInfo;

    class varLeaf_t;

    //---[ Qualifier Info Class ]-----------------
    typedef union {
      typeInfo *type;
      varLeaf_t *varLeaf;
    } typeOrVar;

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

      void free();

      qualifierInfo clone();

      strNode* loadFrom(statement &s,
                        strNode *nodePos);

      int loadFrom(expNode &expRoot,
                   int leafPos);

      //---[ Qualifier Info ]-----------
      bool has(const std::string &qName) const;
      const std::string& get(const int pos) const;

      void add(const std::string &qName,
               int pos = -1);

      void remove(const std::string &qName);
      void remove(const int pos, const int count = 1);

      void clear();
      //================================

      std::string toString() const;
      operator std::string () const;

      friend std::ostream& operator << (std::ostream &out, const qualifierInfo &type);
    };
    //============================================


    //---[ Type Info Class ]----------------------
    class typeInfo {
    public:
      qualifierInfo leftQualifiers;
      std::string name;

      int nestedInfoCount;
      bool *nestedInfoIsType;
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
                   int leafPos);

      int loadTypedefFrom(expNode &expRoot,
                          int leafPos);

      static int delimeterCount(expNode &expRoot,
                                const char *delimiter);

      static int nextDelimeter(expNode &expRoot,
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

      std::string toString(const std::string &tab = "") const;
      operator std::string () const;

      friend std::ostream& operator << (std::ostream &out, const typeInfo &type);
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

      int pointerCount, stackPointerCount;
      expNode *stackExpRoots;

      int argumentCount;
      varInfo *argumentVarInfos;

      int functionNestCount;
      varInfo *functionNests;

      varInfo();

      varInfo(const varInfo &var);
      varInfo& operator = (const varInfo &var);

      varInfo clone();

      static int variablesInStatement(strNode *nodePos);

      //---[ NEW ]------------
      int loadFrom(expNode &expRoot,
                   int leafPos,
                   varInfo *varHasType = NULL);

      int loadTypeFrom(expNode &expRoot,
                       int leafPos,
                       varInfo *varHasType);

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
      //======================

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

      //---[ Variable Info ]------------
      int leftQualifierCount() const;
      int rightQualifierCount() const;

      bool hasQualifier(const std::string &qName) const;

      void addQualifier(const std::string &qName,
                        int pos = -1);

      void addRightQualifier(const std::string &qName,
                             int pos = -1);

      void removeQualifier(const std::string &qName);

      const std::string& getLeftQualifier(const int pos) const;
      const std::string& getRightQualifier(const int pos) const;

      const std::string& getLastLeftQualifier() const;
      const std::string& getLastRightQualifier() const;

      void removeStackPointers();
      //================================

      std::string toString(const bool printType = true) const;
      operator std::string () const;

      friend std::ostream& operator << (std::ostream &out, const varInfo &var);
    };
    //==============================================


    //---[ Kernel Info ]--------------------------
    class kernelInfo {
    public:
      std::string name;
      std::string baseName;

      std::vector<statement*> nestedKernels;
    };
    //==============================================
  };

  class parsedKernelInfo {
  public:
    std::string baseName;
    int nestedKernels;

    inline parsedKernelInfo() :
      baseName(""),
      nestedKernels(0) {}

    inline parsedKernelInfo(parserNamespace::kernelInfo &kInfo) :
      baseName(kInfo.baseName),
      nestedKernels(kInfo.nestedKernels.size()) {}
  };
};

#endif
