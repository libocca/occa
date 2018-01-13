/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
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

#ifndef OCCA_PARSER_TYPES_HEADER
#define OCCA_PARSER_TYPES_HEADER

#include "occa/defines.hpp"
#include "occa/parser/defines.hpp"
#include "occa/parser/nodes.hpp"
#include "occa/tools/json.hpp"

namespace occa {
  class kernelMetadata;

  typedef std::map<std::string, kernelMetadata> kernelMetadataMap;
  typedef kernelMetadataMap::iterator           kernelMetadataMapIterator;
  typedef kernelMetadataMap::const_iterator     cKernelMetadataMapIterator;

  namespace parserNS {
    class expNode;
    class typeInfo;
    class varInfo;

    class varLeaf_t;

    bool isInlinedASM(const std::string &attrName);
    bool isInlinedASM(expNode &expRoot, int leafPos);

    //---[ Scope Info Class ]---------------------
    class scopeInfo {
    public:
      scopeInfo *up;

      std::string name;

      scopeMap_t scopeMap;
      typeMap_t  typeMap;
      varMap_t   varMap;

      scopeInfo();

      inline bool isTheGlobalScope() {
        return (up == NULL);
      }

      void appendVariablesFrom(scopeInfo *scope);

      void add(scopeInfo &scope);
      void add(typeInfo &type);
      void add(varInfo &var);

      scopeInfo* addNamespace(const std::string &namespaceName);

      typeInfo*  hasLocalType(const std::string &typeName);
      varInfo*   hasLocalVariable(const std::string &varName);

      bool removeLocalType(const std::string &typeName);
      bool removeLocalVariable(const std::string &varName);

      bool removeLocalType(typeInfo &type);
      bool removeLocalVariable(varInfo &var);

      void printOnString(std::string &str);

      inline std::string toString() {
        std::string ret;
        printOnString(ret);
        return ret;
      }

      inline operator std::string () {
        std::string ret;
        printOnString(ret);
        return ret;
      }

      friend inline std::ostream& operator << (std::ostream &out, scopeInfo &scope) {
        out << (std::string) scope;

        return out;
      }
    };
    //============================================


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

      friend inline std::ostream& operator << (std::ostream &out, attribute_t &attr) {
        out << (std::string) attr;

        return out;
      }
    };

    void updateAttributeMap(attributeMap_t &attributeMap,
                            const std::string &attrName);

    int updateAttributeMap(attributeMap_t &attributeMap,
                           expNode &expRoot,
                           int leafPos);

    int updateAttributeMapR(attributeMap_t &attributeMap,
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

      inline std::string& operator [] (const int pos) {
        return qualifiers[pos];
      }

      inline const std::string& operator [] (const int pos) const {
        return qualifiers[pos];
      }

      inline int size() {
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

      void printOnString(std::string &str,
                         varInfo *var = NULL);

      inline std::string toString() {
        std::string ret;
        printOnString(ret);
        return ret;
      }

      inline operator std::string () {
        std::string ret;
        printOnString(ret);
        return ret;
      }

      friend inline std::ostream& operator << (std::ostream &out, qualifierInfo &type) {
        out << (std::string) type;

        return out;
      }
    };

    bool expHasQualifier(expNode &allExp, int expPos);
    //============================================


    //---[ Type Info Class ]----------------------
    class typeInfo {
    public:
      scopeInfo *typeScope;

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

      void printOnString(std::string &str,
                         const std::string &tab = "");

      inline std::string toString(const std::string &tab = "") {
        std::string ret;
        printOnString(ret, tab);
        return ret;
      }

      inline operator std::string () {
        std::string ret;
        printOnString(ret);
        return ret;
      }

      friend inline std::ostream& operator << (std::ostream &out, typeInfo &type) {
        out << (std::string) type;

        return out;
      }
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
    }

    class varInfo {
    public:
      scopeInfo *scope;

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
      intVector idxOrdering;

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

      void setupArrayArguments(statement &s);

      varInfo& getArrayArgument(statement &s,
                                varInfo &argVar,
                                const std::string &arrayArgName);

      void setupAttributes();

      void setupDimAttribute();
      void setupArrayArgAttribute();
      void setupIdxOrderAttribute();

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
      void removeAttribute(const std::string &attr);

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

      void printOnString(std::string &str,
                         const bool printType = true);

      inline std::string toString() {
        std::string ret;
        printOnString(ret);
        return ret;
      }

      inline operator std::string () {
        std::string ret;
        printOnString(ret);
        return ret;
      }

      friend inline std::ostream& operator << (std::ostream &out, varInfo &var) {
        out << (std::string) var;

        return out;
      }
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
                          varInfoVector &candidates);
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


    //---[ Kernel Info ]--------------------------
    class argumentInfo {
    public:
      int pos;
      bool isConst;

      argumentInfo();

      argumentInfo(const argumentInfo &info);
      argumentInfo& operator = (const argumentInfo &info);

      static argumentInfo fromJson(const json &j);
      json toJson() const;
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

      occa::kernelMetadata metadata();
    };
    //==============================================
  }

  //---[ Parsed Kernel Info ]---------------------
  typedef parserNS::argumentInfo argumentInfo;

  class kernelMetadata {
  public:
    std::string name, baseName;
    int nestedKernels;

    std::vector<argumentInfo> argumentInfos;

    kernelMetadata();

    kernelMetadata(const kernelMetadata & kInfo);
    kernelMetadata& operator = (const kernelMetadata & kInfo);

    void removeArg(const int pos);

    inline bool argIsConst(const int pos) const {
      if (((size_t) pos) < argumentInfos.size())
        return argumentInfos[pos].isConst;

      return false;
    }

    kernelMetadata getNestedKernelMetadata(const int pos) const;

    static kernelMetadata fromJson(const json &j);
    json toJson() const;
  };
  //==============================================
}

#endif
