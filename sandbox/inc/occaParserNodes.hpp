#ifndef OCCA_PARSER_NODES_HEADER
#define OCCA_PARSER_NODES_HEADER

#include "occaParserDefines.hpp"
#include "occaParserMacro.hpp"

namespace occa {
  namespace parserNamespace {
    //---[ Node ]-----------------------------------
    template <class TM>
    class node {
    public:
      node *left, *right, *up;
      std::vector< node<TM>* > down;
      TM value;

      node();
      node(const TM &t);
      node(const node<TM> &n);

      node& operator = (const node<TM> &n);

      node<TM>* pop();

      node* push(node<TM> *n);
      node* push(const TM &t);

      node* pushDown(node<TM> *n);
      node* pushDown(const TM &t);

      void print(const std::string &tab = "");

      void printPtr(const std::string &tab = "");
    };

    template <class TM>
    void popAndGoRight(node<TM> *&n);

    template <class TM>
    void popAndGoLeft(node<TM> *&n);

    template <class TM>
    node<TM>* firstNode(node<TM> *n);

    template <class TM>
    node<TM>* lastNode(node<TM> *n);

    template <class TM>
    int length(node<TM> *n);
    //==============================================


    //---[ Str Node ]-------------------------------
    class strNode {
    public:
      strNode *left, *right, *up;
      std::vector<strNode*> down;

      std::string value;
      int type, depth, sideDepth;

      int originalLine;

      strNode();
      strNode(const std::string &value_);
      strNode(const strNode &n);

      strNode& operator = (const strNode &n);

      void swapWith(strNode *n);
      void swapWithRight();
      void swapWithLeft();

      void moveLeftOf(strNode *n);
      void moveRightOf(strNode *n);

      strNode* clone() const;

      operator std::string () const;

      strNode* pop();

      strNode* push(strNode *node);
      strNode* push(const std::string &str);

      strNode* pushDown(strNode *node);
      strNode* pushDown(const std::string &str);

      bool hasType(const int type_);

      node<strNode*> getStrNodesWith(const std::string &name_,
                                     const int type_ = everythingType);

      void flatten();

      bool freeLeft();
      bool freeRight();

      void print(const std::string &tab = "");
    };

    std::ostream& operator << (std::ostream &out, const strNode &n);

    void popAndGoRight(strNode *&node);
    void popAndGoLeft(strNode *&node);

    strNode* firstNode(strNode *node);
    strNode* lastNode(strNode *node);

    int length(strNode *node);

    void free(strNode *node);
    //==============================================


    //---[ Exp Node ]-------------------------------
    namespace expType {
      static const int root            = (1 << 0);

      static const int LCR             = (7 << 1);
      static const int L               = (1 << 1);
      static const int C               = (1 << 2);
      static const int R               = (1 << 3);

      static const int qualifier       = (1 << 4);
      static const int type            = (1 << 5);
      static const int presetValue     = (1 << 6);
      static const int variable        = (1 << 7);
      static const int function        = (1 << 8);
      static const int functionPointer = (1 << 9);
    };

    class expNode {
    public:
      std::string value;
      int info;

      expNode *up;

      int leafCount;
      expNode **leaves;
      varInfo *var;
      typeDef *type;

      expNode();

      void loadFromNode(strNode *n);

      void initLoadFromNode(strNode *n);

      void initOrganization();

      void organizeLeaves();

      void organizeLeaves(const int level);

      int mergeRange(const int newLeafType,
                     const int leafPosStart,
                     const int leafPosEnd);

      // [a][::][b]
      void mergeNamespaces();

      int mergeNamespace(const int leafPos);

      // [const] int x
      void mergeQualifiers();

      // [[const] [int] [*]] x
      void mergeTypes();

      // [[[const] [int] [*]] [x]]
      void mergeVariables();

      // 1 [type]                           2 [(]       3 [(]
      // [[qualifiers] [type] [qualifiers]] [(*[name])] [([args])]
      void mergeFunctionPointers();

      // class(...), class{1,2,3}
      void mergeClassConstructs();

      // static_cast<>()
      void mergeCasts();

      // func()
      void mergeFunctionCalls();

      void mergeArguments();

      // (class) x
      void mergeClassCasts();

      // sizeof x
      void mergeSizeOf();

      // new, new [], delete, delete []
      void mergeNewsAndDeletes();

      // throw x
      void mergeThrows();

      // [++]i
      int mergeLeftUnary(const int leafPos);

      // i[++]
      int mergeRightUnary(const int leafPos);

      // a [+] b
      int mergeBinary(const int leafPos);

      // a [?] b : c
      int mergeTernary(const int leafPos);

      //---[ Custom Type Info ]---------
      bool qualifierEndsWithStar() const;

      bool typeEndsWithStar() const;
      //================================

      void freeLeaf(const int leafPos);

      void free();

      void print(const std::string &tab = "");

      operator std::string () const;

      friend std::ostream& operator << (std::ostream &out, const expNode &n);
    };
    //==============================================
  };
};

#include "occaParserNodes.tpp"

#endif
