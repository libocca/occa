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

#include "variable.hpp"
#include "builtins/transforms/finders.hpp"

namespace occa {
  namespace lang {
    namespace transforms {
      //---[ Statement ]----------------
      statementFinder::statementFinder() :
        statementTransform() {}

      void statementFinder::getStatements(statement_t &smnt,
                                          statementPtrVector &statements_) {
        statements = &statements_;
        apply(smnt);
      }

      statement_t* statementFinder::transformStatement(statement_t &smnt) {
        if (matchesStatement(smnt)) {
          statements->push_back(&smnt);
        }
        return &smnt;
      }

      statementTypeFinder::statementTypeFinder(const int validStatementTypes_) {
        validStatementTypes = validStatementTypes_;
      }

      bool statementTypeFinder::matchesStatement(statement_t &smnt) {
        return true;
      }

      statementAttrFinder::statementAttrFinder(const int validStatementTypes_,
                                               const std::string &attr_) :
        attr(attr_) {
        validStatementTypes = validStatementTypes_;
      }

      bool statementAttrFinder::matchesStatement(statement_t &smnt) {
        return smnt.hasAttribute(attr);
      }

      statementMatcherFinder::statementMatcherFinder(const int validStatementTypes_,
                                                     statementMatcher matcher_) :
        matcher(matcher_) {
        validStatementTypes = validStatementTypes_;
      }

      bool statementMatcherFinder::matchesStatement(statement_t &smnt) {
        return matcher(smnt);
      }
      //================================

      //---[ Expr Node ]----------------
      exprNodeFinder::exprNodeFinder() {}

      void exprNodeFinder::getExprNodes(exprNode &expr,
                                        exprNodeVector &exprNodes_) {
        exprNodes = &exprNodes_;
        apply(expr);
      }

      exprNode* exprNodeFinder::transformExprNode(exprNode &expr) {
        if (matchesExprNode(expr)) {
          exprNodes->push_back(&expr);
        }
        return &expr;
      }

      exprNodeTypeFinder::exprNodeTypeFinder(const int validExprNodeTypes_) {
        validExprNodeTypes = validExprNodeTypes_;
      }

      bool exprNodeTypeFinder::matchesExprNode(exprNode &expr) {
        return true;
      }

      exprNodeAttrFinder::exprNodeAttrFinder(const int validExprNodeTypes_,
                                             const std::string &attr_) :
        attr(attr_) {
        validExprNodeTypes = (validExprNodeTypes_
                              & (exprNodeType::type     |
                                 exprNodeType::variable |
                                 exprNodeType::function));
      }

      bool exprNodeAttrFinder::matchesExprNode(exprNode &expr) {
        return expr.hasAttribute(attr);
      }

      exprNodeMatcherFinder::exprNodeMatcherFinder(const int validExprNodeTypes_,
                                                   exprNodeMatcher matcher_) :
        matcher(matcher_) {
        validExprNodeTypes = validExprNodeTypes_;
      }

      bool exprNodeMatcherFinder::matchesExprNode(exprNode &expr) {
        return matcher(expr);
      }
      //================================

      //---[ Statement + Expr ]---------
      statementExprFinder::statementExprFinder(const int validExprNodeTypes_,
                                               exprNodeMatcher matcher_) :
        statementTypeFinder(statementType::declaration |
                            statementType::expression),
        exprNodeMatcherFinder(validExprNodeTypes_,
                              matcher_) {

      }

      void statementExprFinder::getExprNodes(statement_t &smnt,
                                             statementExprMap &exprMap) {
        statementPtrVector statements_;
        getStatements(smnt, statements_);

        const int smntCount = (int) statements_.size();
        for (int i = 0; i < smntCount; ++i) {
          statement_t &foundSmnt = *(statements_[i]);
          exprNodeVector exprNodes_;
          bool addSmnt = false;

          // Expression
          if (foundSmnt.type() & statementType::expression) {
            expressionStatement &exprSmnt = (expressionStatement&) foundSmnt;
            exprNodeFinder::getExprNodes(*(exprSmnt.expr),
                                         exprNodes_);
          } else {
            // Declaration
            declarationStatement &declSmnt = (declarationStatement&) foundSmnt;
            const int declCount = (int) declSmnt.declarations.size();
            for (int di = 0; di < declCount; ++di) {
              variableDeclaration &decl = declSmnt.declarations[di];
              variableNode varNode(decl.variable->source,
                                   *decl.variable);
              if (matcher(varNode)) {
                addSmnt = true;
              }
              if (decl.value) {
                exprNodeFinder::getExprNodes(*(decl.value),
                                             exprNodes_);
              }
            }
          }

          if (exprNodes_.size()
              || addSmnt) {
            exprMap[&foundSmnt] = exprNodes_;
          }
        }
      }
      //================================

      //---[ Statement Tree ]-----------
      smntTreeNode::smntTreeNode(statement_t *smnt_) :
        smnt(smnt_) {}

      smntTreeNode::~smntTreeNode() {
        const int count = (int) children.size();
        for (int i = 0; i < count; ++i) {
          delete children[i];
        }
      }

      void smntTreeNode::free() {
        const int count = (int) children.size();
        for (int i = 0; i < count; ++i) {
          delete children[i];
        }
        children.clear();
      }

      int smntTreeNode::size() {
        return (int) children.size();
      }

      smntTreeNode* smntTreeNode::operator [] (const int index) {
        const int count = (int) children.size();
        if ((0 <= index) && (index < count)) {
          return children[index];
        }
        return NULL;
      }

      void smntTreeNode::add(smntTreeNode *node) {
        children.push_back(node);
      }

      smntTreeHistory::smntTreeHistory(smntTreeNode *node_,
                                       statement_t *smnt_) :
        node(node_),
        smnt(smnt_) {}

      smntTreeFinder::smntTreeFinder(const int validStatementTypes_,
                                     statement_t &smnt,
                                     smntTreeNode &root_,
                                     statementMatcher matcher_) :
        root(root_),
        matcher(matcher_),
        validSmntTypes(validStatementTypes_) {

        downToUp = false;
        validStatementTypes = statementType::all;

        // Fill the whole ancestry with NULL
        // Every statement comes from smnt so it's safe
        history.push_front(
          smntTreeHistory(&root, smnt.up)
        );
        statement_t *s = &smnt;
        while (s->up) {
          history.push_front(
            smntTreeHistory(NULL, s->up->up)
          );
          s = s->up;
        }
      }

      statement_t* smntTreeFinder::transformStatement(statement_t &smnt) {
        updateHistory(smnt);

        smntTreeNode *node = NULL;
        if ((smnt.type() & validSmntTypes)
            && matchesStatement(smnt)) {
          node = new smntTreeNode(&smnt);
          addNode(*node);
        }
        history.push_back(
          smntTreeHistory(node, &smnt)
        );

        return &smnt;
      }

      bool smntTreeFinder::matchesStatement(statement_t &smnt) {
        return matcher(smnt);
      }

      void smntTreeFinder::updateHistory(statement_t &smnt) {
        statementPtrList path;
        getStatementPath(smnt, path);

        statementPtrList::iterator pathIt = path.begin();
        smntTreeHistoryList::iterator historyIt = history.begin();
        while ((pathIt != path.end())
               && (historyIt != history.end())) {
          statement_t *pathSmnt = *pathIt;
          statement_t *historySmnt = historyIt->smnt;
          if (pathSmnt != historySmnt) {
            break;
          }
          ++pathIt;
          ++historyIt;
        }
        // Erase forked path
        history.erase(historyIt, history.end());
      }

      void smntTreeFinder::getStatementPath(statement_t &smnt,
                                            statementPtrList &path) {
        statement_t *node = smnt.up;
        while (node) {
          path.push_front(node);
          node = node->up;
        }
        // Add root's parent
        path.push_front(NULL);
      }

      void smntTreeFinder::addNode(smntTreeNode &node) {
        // Add to first real node in our ancestry path
        smntTreeHistoryList::reverse_iterator it = history.rbegin();
        while (it != history.rend()) {
          if (it->node) {
            it->node->add(&node);
            break;
          }
          ++it;
        }
      }
      //================================
    }
    //---[ Helper Methods ]-------------
    void findStatements(const int validStatementTypes,
                        statement_t &smnt,
                        statementMatcher matcher,
                        statementPtrVector &statements) {

      transforms::statementMatcherFinder finder(validStatementTypes,
                                                matcher);
      finder.getStatements(smnt, statements);
    }

    void findStatements(const int validExprNodeTypes,
                        statement_t &smnt,
                        exprNodeMatcher matcher,
                        statementExprMap &exprMap) {
      transforms::statementExprFinder finder(validExprNodeTypes,
                                             matcher);
      finder.getExprNodes(smnt, exprMap);
    }

    void findStatementsByType(const int validStatementTypes,
                              statement_t &smnt,
                              statementPtrVector &statements) {

      transforms::statementTypeFinder finder(validStatementTypes);
      finder.getStatements(smnt, statements);
    }

    void findStatementsByAttr(const int validStatementTypes,
                              const std::string &attr,
                              statement_t &smnt,
                              statementPtrVector &statements) {

      transforms::statementAttrFinder finder(validStatementTypes, attr);
      finder.getStatements(smnt, statements);
    }

    void findExprNodes(const int validExprNodeTypes,
                       exprNode &expr,
                       exprNodeMatcher matcher,
                       exprNodeVector &exprNodes) {

      transforms::exprNodeMatcherFinder finder(validExprNodeTypes,
                                               matcher);
      finder.getExprNodes(expr, exprNodes);
    }

    void findExprNodesByType(const int validExprNodeTypes,
                             exprNode &expr,
                             exprNodeVector &exprNodes) {

      transforms::exprNodeTypeFinder finder(validExprNodeTypes);
      finder.getExprNodes(expr, exprNodes);
    }

    void findExprNodesByAttr(const int validExprNodeTypes,
                             const std::string &attr,
                             exprNode &expr,
                             exprNodeVector &exprNodes) {

      transforms::exprNodeAttrFinder finder(validExprNodeTypes, attr);
      finder.getExprNodes(expr, exprNodes);
    }

    void findStatementTree(const int validStatementTypes,
                           statement_t &smnt,
                           statementMatcher matcher,
                           transforms::smntTreeNode &root) {
      transforms::smntTreeFinder finder(validStatementTypes,
                                        smnt,
                                        root,
                                        matcher);
      finder.apply(smnt);
    }
    //==================================
  }
}
