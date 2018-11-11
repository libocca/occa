#ifndef OCCA_LANG_BUILTINS_TRANSFORMS_STATEMENTTREE_HEADER
#define OCCA_LANG_BUILTINS_TRANSFORMS_STATEMENTTREE_HEADER

#include <vector>

#include <occa/lang/exprNode.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/statementTransform.hpp>
#include <occa/lang/exprTransform.hpp>

namespace occa {
  namespace lang {
    typedef std::vector<statement_t*> statementPtrVector;
    typedef std::vector<exprNode*>    exprNodeVector;

    namespace transforms {
      //---[ Statement ]----------------
      class statementTreeFinder : public statementTransform {
      //---[ Statement Node ]-----------
      class smntNode;

      typedef std::vector<statementNode*> statementNodeVector;

      class statementNode {
      public:
        statement_t *smnt;
        statementNodeVector children;

        statementNode(statement_t *smnt_ = NULL);

        int size();
        statementNode* operator [] (const int index);
      };

      class statementTree {
      public:
        statementNode root;

        statementTree(statementPtrVector &statements);

        void getAncestry(statement_t *smnt,
                         statementPath &path);
      };
      //================================