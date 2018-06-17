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

#include "../../parserUtils.hpp"
#include <occa/lang/builtins/transforms/finders.hpp>

void testStatementFinder();
void testExprNodeFinder();
void testStatementExprFinder();
void testStatementTreeFinder();

int main(const int argc, const char **argv) {
  parser.addAttribute<dummy>();
  parser.addAttribute<attributes::kernel>();
  parser.addAttribute<attributes::outer>();
  parser.addAttribute<attributes::inner>();
  parser.addAttribute<attributes::shared>();
  parser.addAttribute<attributes::exclusive>();

  testStatementFinder();
  testExprNodeFinder();
  testStatementExprFinder();
  testStatementTreeFinder();

  return 0;
}

void testStatementFinder() {
  statementPtrVector statements;
  parseSource(
    "@dummy int d1;\n"
    "@dummy void f1();\n"
    "@dummy void f2();\n"
    "@dummy void f3();\n"
    "@dummy void fd1() {}\n"
    "@dummy void fd2() {\n"
    "  @dummy int d2;\n"
    "  @dummy e1 + 1;\n"
    "  @dummy e2 += 1;\n"
    "  @dummy {\n"
    "    @dummy {}\n"
    "    @dummy int d3;\n"
    "    @dummy int d4;\n"
    "    @dummy e3++;\n"
    "    @dummy e4--;\n"
    "    @dummy e5 *= 1;\n"
    "  }\n"
    "}\n"
  );

  findStatementsByAttr(statementType::block,
                       "dummy",
                       parser.root,
                       statements);
  ASSERT_EQ(2,
            (int) statements.size());
  statements.clear();

  findStatementsByAttr(statementType::expression,
                       "dummy",
                       parser.root,
                       statements);
  ASSERT_EQ(5,
            (int) statements.size());
  statements.clear();

  findStatementsByAttr(statementType::declaration,
                       "dummy",
                       parser.root,
                       statements);
  ASSERT_EQ(4,
            (int) statements.size());
  statements.clear();

  findStatementsByAttr(statementType::function,
                       "dummy",
                       parser.root,
                       statements);
  ASSERT_EQ(3,
            (int) statements.size());
  statements.clear();

  findStatementsByAttr(statementType::functionDecl,
                       "dummy",
                       parser.root,
                       statements);
  ASSERT_EQ(2,
            (int) statements.size());
  statements.clear();
}


void testExprNodeFinder() {
  exprNodeVector exprNodes;
  parseSource(
    "@dummy typedef int t1;\n"
    "@dummy int foo() {}\n"
    "@dummy int bar() {}\n"
    "@dummy int d1, d2, d3;\n"
    "(t1) (d1 * d2 * d3 + foo() + bar());\n"
  );

#define exprRoot (*(((expressionStatement*) parser.root.children[4])->expr))

  // TODO 1.1: Deal with non-node values inside exprNodes
  //           For example, parenCast has a vartype_t
  // findExprNodesByAttr(exprNodeType::type,
  //                     "dummy",
  //                     exprRoot,
  //                     exprNodes);
  // ASSERT_EQ(1,
  //                   (int) exprNodes.size());
  // exprNodes.clear();

  findExprNodesByAttr(exprNodeType::variable,
                      "dummy",
                      exprRoot,
                      exprNodes);
  ASSERT_EQ(3,
            (int) exprNodes.size());
  exprNodes.clear();

  findExprNodesByAttr(exprNodeType::function,
                      "dummy",
                      exprRoot,
                      exprNodes);
  ASSERT_EQ(2,
            (int) exprNodes.size());
  exprNodes.clear();

#undef exprRoot
}

bool exclusiveMatcher(exprNode &expr) {
  variable_t &var = ((variableNode&) expr).value;
  return var.hasAttribute("exclusive");
}

void testStatementExprFinder() {
  parseSource(
    "@exclusive int x;\n"
    "for (int i = 0; i < N; ++i) {\n"
    "  x = (x + x + x) * x;\n"
    "  @exclusive int y = x, z;\n"
    "}"
  );
  statementExprMap exprMap;
  findStatements(exprNodeType::variable,
                 parser.root,
                 exclusiveMatcher,
                 exprMap);

  ASSERT_EQ(3,
            (int) exprMap.size());

  statementExprMap::iterator it = exprMap.begin();
  while (it != exprMap.end()) {
    statement_t *smnt = it->first;
    exprNodeVector *exprNodes = &(it->second);

    if (smnt->type() & statementType::expression) {
      // x = (x + x + x) * x
      ASSERT_EQ(5,
                (int) exprNodes->size());
    } else {
      declarationStatement &declSmnt = *((declarationStatement*) smnt);
      if (declSmnt.declarations.size() == 1) {
        // x; (x doesn't 'count' since it's not in an variableNode)
        ASSERT_EQ(0,
                  (int) exprNodes->size());
      } else {
        // y = x, z; (y doesn't 'count' since it's not in an variableNode)
        ASSERT_EQ(1,
                  (int) exprNodes->size());
      }
    }

    ++it;
  }
}

bool blockMatcher(statement_t &smnt) {
  return true;
}

void testStatementTreeFinder() {
  parseSource(
    "{ while(true) {\n"
    "  {}\n"
    "  {\n"
    "    if(true){{}{}{}}\n"
    "  }\n"
    "  {}\n"
    "}}"
  );
  transforms::smntTreeNode root;
  findStatementTree(statementType::block,
                    parser.root,
                    blockMatcher,
                    root);

  ASSERT_EQ(1,
            root.size());

  transforms::smntTreeNode &r0 = *(root[0]);
  ASSERT_EQ(1,
            r0.size());

  transforms::smntTreeNode &r00 = *(r0[0]);
  ASSERT_EQ(3,
            r00.size());

  ASSERT_EQ(0,
            r00[0]->size());
  ASSERT_EQ(3,
            r00[1]->size());
  ASSERT_EQ(0,
            r00[2]->size());

  transforms::smntTreeNode &r001 = *(r00[1]);
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(0,
              r001[i]->size());
  }
  root.free();
}
