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

#ifndef OCCA_LANG_BUILTINS_TRANSFORMS_FINDERS_HEADER
#define OCCA_LANG_BUILTINS_TRANSFORMS_FINDERS_HEADER

#include <map>
#include <list>
#include <vector>

#include <occa/lang/exprNode.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/statementTransform.hpp>
#include <occa/lang/exprTransform.hpp>

namespace occa {
  namespace lang {
    typedef std::vector<statement_t*>              statementPtrVector;
    typedef std::vector<exprNode*>                 exprNodeVector;
    typedef std::map<statement_t*, exprNodeVector> statementExprMap;

    typedef bool (*statementMatcher)(statement_t &smnt);
    typedef bool (*exprNodeMatcher)(exprNode &expr);
    typedef exprNode* (*smntExprTransform)(statement_t &smnt,
                                           exprNode &expr,
                                           const bool isBeingDeclared);

    namespace transforms {
      //---[ Statement ]----------------
      class statementFinder : public statementTransform {
      private:
        statementPtrVector *statements;

      public:
        statementFinder();

        void getStatements(statement_t &smnt,
                           statementPtrVector &statements_);

        virtual statement_t* transformStatement(statement_t &smnt);

        virtual bool matchesStatement(statement_t &smnt) = 0;
      };

      class statementTypeFinder : public statementFinder {
      public:
        statementTypeFinder(const int validStatementTypes_);

        virtual bool matchesStatement(statement_t &smnt);
      };

      class statementAttrFinder : public statementFinder {
      private:
        std::string attr;

      public:
        statementAttrFinder(const int validStatementTypes_,
                            const std::string &attr_);

        virtual bool matchesStatement(statement_t &smnt);
      };

      class statementMatcherFinder : public statementFinder {
      public:
        statementMatcher matcher;

        statementMatcherFinder(const int validStatementTypes_,
                               statementMatcher matcher_);

        virtual bool matchesStatement(statement_t &smnt);
      };
      //================================

      //---[ Expr Node ]----------------
      class exprNodeFinder : public exprTransform {
      private:
        exprNodeVector *exprNodes;

      public:
        exprNodeFinder();

        void getExprNodes(exprNode &node,
                          exprNodeVector &exprNodes_);

        virtual exprNode* transformExprNode(exprNode &expr);

        virtual bool matchesExprNode(exprNode &expr) = 0;
      };

      class exprNodeTypeFinder : public exprNodeFinder {
      public:
        exprNodeTypeFinder(const int validExprNodeTypes_);

        virtual bool matchesExprNode(exprNode &expr);
      };

      class exprNodeAttrFinder : public exprNodeFinder {
      private:
        std::string attr;

      public:
        exprNodeAttrFinder(const int validExprNodeTypes_,
                           const std::string &attr_);

        virtual bool matchesExprNode(exprNode &expr);
      };

      class exprNodeMatcherFinder : public exprNodeFinder {
      public:
        exprNodeMatcher matcher;

        exprNodeMatcherFinder(const int validExprNodeTypes_,
                              exprNodeMatcher matcher_);

        virtual bool matchesExprNode(exprNode &expr);
      };
      //================================

      //---[ Statement + Expr ]---------
      class statementExprTransform : public statementTransform,
                                     public exprTransform {
      public:
        smntExprTransform transform;
        statement_t *currentSmnt;
        bool nextExprIsBeingDeclared;

        statementExprTransform(const int validExprNodeTypes_,
                               smntExprTransform transform_ = NULL);

        statementExprTransform(const int validStatementTypes_,
                               const int validExprNodeTypes_,
                               smntExprTransform transform_ = NULL);

        virtual statement_t* transformStatement(statement_t &smnt);

        virtual exprNode* transformExprNode(exprNode &node);
      };

      class statementExprFinder : public statementTypeFinder,
                                  public exprNodeMatcherFinder {
      public:
        statementExprFinder(const int validExprNodeTypes_,
                            exprNodeMatcher matcher_);

        statementExprFinder(const int validStatementTypes_,
                            const int validExprNodeTypes_,
                            exprNodeMatcher matcher_);

        void getExprNodes(statement_t &smnt,
                          statementExprMap &exprMap);
      };
      //================================

      //---[ Statement Tree ]-----------
      class smntTreeNode;
      class smntTreeHistory;

      typedef std::list<statement_t*>    statementPtrList;
      typedef std::vector<smntTreeNode*> smntTreeNodeVector;
      typedef std::list<smntTreeHistory> smntTreeHistoryList;

      class smntTreeNode {
      public:
        statement_t *smnt;
        smntTreeNodeVector children;

        smntTreeNode(statement_t *smnt_ = NULL);
        ~smntTreeNode();

        void free();

        int size();
        smntTreeNode* operator [] (const int index);

        void add(smntTreeNode *node);
      };

      class smntTreeHistory {
      public:
        smntTreeNode *node;
        statement_t *smnt;

        smntTreeHistory(smntTreeNode *node_,
                        statement_t *smnt_);
      };

      class smntTreeFinder : public statementTransform {
      public:
        smntTreeNode &root;
        statementMatcher matcher;
        smntTreeHistoryList history;
        int validSmntTypes;

        smntTreeFinder(const int validStatementTypes_,
                       statement_t &smnt,
                       smntTreeNode &root_,
                       statementMatcher matcher_);

        virtual statement_t* transformStatement(statement_t &smnt);

        virtual bool matchesStatement(statement_t &smnt);

        void updateHistory(statement_t &smnt);

        void getStatementPath(statement_t &smnt,
                              statementPtrList &path);

        void addNode(smntTreeNode &node);
      };
      //================================
    }

    //---[ Helper Methods ]-------------
    void transformExprNodes(const int validExprNodeTypes,
                            statement_t &smnt,
                            smntExprTransform transform);

    void findStatements(const int validStatementTypes,
                        statement_t &smnt,
                        statementMatcher matcher,
                        statementPtrVector &statements);

    void findStatements(const int validExprNodeTypes,
                        statement_t &smnt,
                        exprNodeMatcher matcher,
                        statementExprMap &exprMap);


    void findStatements(const int validStatementTypes,
                        const int validExprNodeTypes,
                        statement_t &smnt,
                        exprNodeMatcher matcher,
                        statementExprMap &exprMap);

    void findStatementsByType(const int validStatementTypes,
                              statement_t &smnt,
                              statementPtrVector &statements);

    void findStatementsByAttr(const int validStatementTypes,
                              const std::string &attr,
                              statement_t &smnt,
                              statementPtrVector &statements);

    void findExprNodes(const int validExprNodeTypes,
                       exprNode &expr,
                       exprNodeMatcher matcher,
                       exprNodeVector &exprNodes);

    void findExprNodesByType(const int validExprNodeTypes,
                             exprNode &expr,
                             exprNodeVector &exprNodes);

    void findExprNodesByAttr(const int validExprNodeTypes,
                             const std::string &attr,
                             exprNode &expr,
                             exprNodeVector &exprNodes);

    void findStatementTree(const int validStatementTypes,
                           statement_t &smnt,
                           statementMatcher matcher,
                           transforms::smntTreeNode &root);
    //==================================
  }
}

#endif
