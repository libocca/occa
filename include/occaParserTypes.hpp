#ifndef OCCA_PARSER_TYPES_HEADER
#define OCCA_PARSER_TYPES_HEADER

#include "occaParserDefines.hpp"
#include "occaParserNodes.hpp"
#include "occaParserStatement.hpp"

namespace occa {
  namespace parserNamespace {
    //---[ Type Definitions ]-----------------------
    /*
      | struct {          |  members    = {float x, float y, float z}
      |   union {         |  allMembers = {union{float x, float y}, float z}
      |     float x;      |
      |     float y;      |
      |   }               |
      |   float z;        |
      | }                 |
      |                   |
      | int (*f)(void *a) | allMembers = {int, void* a}
    */
    class typeDef {
    public:
      typeDef *up;

      std::string typeName, varName;
      int typeInfo;

      int pointerCount;
      std::vector<std::string> stackPointerSizes;

      int bitField;

      scopeTypeMap_t memberTypes;
      scopeVarMap_t  memberVars;

      std::vector<void*> allMembers;
      std::vector<char> memberInfo;

      typeDef *typedefing, *typedefingBase;
      bool typedefUsesName;

      typeDef();

      void addVar(varInfo *def);

      void addType(typeDef *def);

      typeDef& addType(const std::string &newVarName);

      void loadFromNode(statement &s,
                        strNode *&n);

      void loadStructPartsFromNode(statement &s, strNode *n);
      void loadEnumPartsFromNode(statement &s, strNode *n);

      std::string print(const std::string &tab = "", const int printStyle = 0) const;

      operator std::string() const;
    };

    std::ostream& operator << (std::ostream &out, const typeDef &def);
    //==============================================


    //---[ New Variable Info ]----------------------
    class expNode;

    class _qualifierInfo {
    public:
      int qualifierCount;
      std::string *qualifiers;

      _qualifierInfo();

      _qualifierInfo(const _qualifierInfo &q);
      _qualifierInfo& operator = (const _qualifierInfo &q);

      _qualifierInfo clone();

      strNode* loadFrom(statement &s,
                        strNode *nodePos);

      std::string toString();
    };

    class _typeInfo {

    };

    namespace _varType {
      static const int var             = (1 << 0);

      static const int functionType    = (3 << 1);
      static const int function        = (1 << 1);
      static const int functionPointer = (1 << 2);
    };

    class _varInfo {
    public:
      int info;

      _qualifierInfo leftQualifiers, rightQualifiers;

      typeDef *baseType;

      std::string name;

      int pointerCount, stackPointerCount;
      expNode *stackExpRoots;

      int argumentCount;
      _varInfo *argumentVarInfos;

      _varInfo *nestedFunctionPointer;

      _varInfo();

      _varInfo(const _varInfo &var);
      _varInfo& operator = (const _varInfo &var);

      static int variablesInStatement(strNode *nodePos);

      strNode* loadValueFrom(statement &s,
                             strNode *nodePos,
                             _varInfo *varHasType = NULL);

      strNode* loadTypeFrom(statement &s,
                            strNode *nodePos,
                            _varInfo *varHasType);

      int getVarInfoFrom(statement &s,
                         strNode *nodePos);

      strNode* loadNameFrom(statement &s,
                            strNode *nodePos);

      strNode* loadStackPointersFrom(statement &s,
                                     strNode *nodePos);

      strNode* loadArgsFrom(statement &s,
                            strNode *nodePos);

      std::string toString(const bool printType = true);

      operator std::string ();

      friend std::ostream& operator << (std::ostream &out, _varInfo &var);
    };
    //==============================================


    //---[ Variable Info ]--------------------------
    class varInfo {
    public:
      typeDef *type;
      std::string altType, name;
      int typeInfo;

      int bitField;

      int pointerCount;
      std::vector<std::string> descriptors;
      std::vector<std::string> stackPointerSizes;

      // Function {Proto, Def | Ptr}
      //    { arg1 , arg2 , ... }
      std::vector<varInfo*> vars;

      std::vector<std::string> extraInfo;

      varInfo();
      varInfo(const varInfo &vi);

      varInfo& operator = (const varInfo &vi);

      std::string postTypeStr() const;

      std::string decoratedType() const;

      int hasDescriptor(const std::string descriptor) const;

      void removeDescriptor(const std::string descriptor);
      void removeDescriptor(const int descPos);

      strNode* makeStrNodeChain(const int depth     = 0,
                                const int sideDepth = 0) const;

      operator std::string() const;

      std::string podString() const;

      std::string functionString() const;

      std::string functionPointerString() const;
    };

    std::ostream& operator << (std::ostream &out, const varInfo &info);
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
