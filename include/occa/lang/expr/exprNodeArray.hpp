#ifndef OCCA_LANG_EXPRNODE_EXPRNODEARRAY_HEADER
#define OCCA_LANG_EXPRNODE_EXPRNODEARRAY_HEADER

#include <occa/lang/utils/array.hpp>

namespace occa {
  namespace lang {
    class statement_t;
    class exprNode;
    class smntExprNode;

    typedef std::function<exprNode* (smntExprNode smntExpr)> smntExprMapCallback;
    typedef std::function<bool (smntExprNode smntExpr)> smntExprFilterCallback;
    typedef std::function<void (smntExprNode smntExpr)> smntExprVoidCallback;
    typedef std::function<void (smntExprNode smntExpr, exprNode **nodeRef)> smntExprWithRefVoidCallback;

    class smntExprNode {
     public:
      statement_t *smnt;
      exprNode *node;
      exprNode *rootNode;

      inline smntExprNode() :
          smnt(NULL),
          node(NULL),
          rootNode(NULL) {}

      inline smntExprNode(statement_t *_smnt,
                          exprNode *_node) :
          smnt(_smnt),
          node(_node),
          rootNode(_node) {}

      inline smntExprNode(statement_t *_smnt,
                          exprNode *_node,
                          exprNode *_rootNode) :
          smnt(_smnt),
          node(_node),
          rootNode(_rootNode) {}
    };

    class exprNodeArray : public array<smntExprNode> {
     public:
      OCCA_LANG_ARRAY_DEFINE_METHODS(exprNodeArray, smntExprNode)

      static exprNodeArray from(statement_t *smnt, exprNode *node);

      void inplaceMap(smntExprMapCallback func) const;

      void flatInplaceMap(smntExprMapCallback func) const;

      exprNodeArray flatFilter(smntExprFilterCallback func) const;

      void nestedForEach(smntExprVoidCallback func) const;

     private:
      void forEachWithRef(smntExprWithRefVoidCallback func) const;
      void nestedForEachWithRef(smntExprWithRefVoidCallback func) const;

     public:
      // Filter helper functions
      exprNodeArray flatFilterByExprType(const int allowedExprNodeType) const;

      exprNodeArray flatFilterByExprType(const int allowedExprNodeType, const std::string &attr) const;
    };
  }
}

#endif
