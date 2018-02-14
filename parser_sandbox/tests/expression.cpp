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
#include "occa/tools/env.hpp"
#include "occa/tools/testing.hpp"

#include "typeBuiltins.hpp"
#include "expression.hpp"
#include "tokenizer.hpp"

void testOperatorNodes();
void testOtherNodes();
void testPairMatching();
void testCanEvaluate();
void testEval();

int main(const int argc, const char **argv) {
  testOperatorNodes();
  testOtherNodes();
  testPairMatching();
  testCanEvaluate();
  testEval();

  return 0;
}

using namespace occa;
using namespace occa::lang;

void testOperatorNodes() {
  primitiveNode a(NULL, 1);
  primitiveNode b(NULL, 2.0);
  primitiveNode c(NULL, false);

  std::cout
    //---[ Left Unary ]---------------
    << "not_           : " << leftUnaryOpNode(NULL, op::not_, a).toString() << '\n'
    << "positive       : " << leftUnaryOpNode(NULL, op::positive, a).toString() << '\n'
    << "negative       : " << leftUnaryOpNode(NULL, op::negative, a).toString() << '\n'
    << "tilde          : " << leftUnaryOpNode(NULL, op::tilde, a).toString() << '\n'
    << "leftIncrement  : " << leftUnaryOpNode(NULL, op::leftIncrement, a).toString() << '\n'
    << "leftDecrement  : " << leftUnaryOpNode(NULL, op::leftDecrement, a).toString() << '\n'
    //================================

    //---[ Right Unary ]--------------
    << "rightIncrement : " << rightUnaryOpNode(NULL, op::rightIncrement, a).toString() << '\n'
    << "rightDecrement : " << rightUnaryOpNode(NULL, op::rightDecrement, a).toString() << '\n'
    //================================

    //---[ Binary ]-------------------
    << "add            : " << binaryOpNode(NULL, op::add, a, b).toString() << '\n'
    << "sub            : " << binaryOpNode(NULL, op::sub, a, b).toString() << '\n'
    << "mult           : " << binaryOpNode(NULL, op::mult, a, b).toString() << '\n'
    << "div            : " << binaryOpNode(NULL, op::div, a, b).toString() << '\n'
    << "mod            : " << binaryOpNode(NULL, op::mod, a, b).toString() << '\n'
    << "lessThan       : " << binaryOpNode(NULL, op::lessThan, a, b).toString() << '\n'
    << "lessThanEq     : " << binaryOpNode(NULL, op::lessThanEq, a, b).toString() << '\n'
    << "equal          : " << binaryOpNode(NULL, op::equal, a, b).toString() << '\n'
    << "notEqual       : " << binaryOpNode(NULL, op::notEqual, a, b).toString() << '\n'
    << "greaterThan    : " << binaryOpNode(NULL, op::greaterThan, a, b).toString() << '\n'
    << "greaterThanEq  : " << binaryOpNode(NULL, op::greaterThanEq, a, b).toString() << '\n'
    << "and_           : " << binaryOpNode(NULL, op::and_, a, b).toString() << '\n'
    << "or_            : " << binaryOpNode(NULL, op::or_, a, b).toString() << '\n'
    << "bitAnd         : " << binaryOpNode(NULL, op::bitAnd, a, b).toString() << '\n'
    << "bitOr          : " << binaryOpNode(NULL, op::bitOr, a, b).toString() << '\n'
    << "xor_           : " << binaryOpNode(NULL, op::xor_, a, b).toString() << '\n'
    << "leftShift      : " << binaryOpNode(NULL, op::leftShift, a, b).toString() << '\n'
    << "rightShift     : " << binaryOpNode(NULL, op::rightShift, a, b).toString() << '\n'
    << "addEq          : " << binaryOpNode(NULL, op::addEq, a, b).toString() << '\n'
    << "subEq          : " << binaryOpNode(NULL, op::subEq, a, b).toString() << '\n'
    << "multEq         : " << binaryOpNode(NULL, op::multEq, a, b).toString() << '\n'
    << "divEq          : " << binaryOpNode(NULL, op::divEq, a, b).toString() << '\n'
    << "modEq          : " << binaryOpNode(NULL, op::modEq, a, b).toString() << '\n'
    << "andEq          : " << binaryOpNode(NULL, op::andEq, a, b).toString() << '\n'
    << "orEq           : " << binaryOpNode(NULL, op::orEq, a, b).toString() << '\n'
    << "xorEq          : " << binaryOpNode(NULL, op::xorEq, a, b).toString() << '\n'
    << "leftShiftEq    : " << binaryOpNode(NULL, op::leftShiftEq, a, b).toString() << '\n'
    << "rightShiftEq   : " << binaryOpNode(NULL, op::rightShiftEq, a, b).toString() << '\n'
    //================================

    //---[ Ternary ]------------------
    << "ternary        : " << ternaryOpNode(NULL, a, b, c).toString() << '\n'
    //================================
    ;
}

void testOtherNodes() {
  qualifiers_t q1;
  q1.add(volatile_);

  type_t t1_0(float_);
  t1_0.addQualifier(const_);
  pointerType t1_1(const_, t1_0);
  pointerType t1(t1_1);

  variable var_(t1, "var");
  variableNode var(NULL, var_);

  primitiveNode one(NULL, 1);
  primitiveNode two(NULL, 2);
  primitiveNode three(NULL, 3);

  exprNodeVector args;
  args.push_back(&one);
  args.push_back(&two);
  args.push_back(&three);

  std::cout << "one                 : " << one.toString() << '\n'
            << "var                 : " << var.toString() << '\n'
            << "subscript           : " << subscriptNode(NULL, var, one).toString() << '\n'
            << "callNode            : " << callNode(NULL, var, args).toString() << '\n'
            << "newNode             : " << newNode(NULL, t1, var, three).toString() << '\n'
            << "newNode             : " << newNode(NULL, t1, var).toString() << '\n'
            << "deleteNode          : " << deleteNode(NULL, var, false).toString() << '\n'
            << "deleteNode          : " << deleteNode(NULL, var, true).toString() << '\n'
            << "throwNode           : " << throwNode(NULL, one).toString() << '\n'
            << "sizeofNode          : " << sizeofNode(NULL, var).toString() << '\n'
            << "funcCastNode        : " << funcCastNode(NULL, t1, var).toString() << '\n'
            << "parenCastNode       : " << parenCastNode(NULL, t1, var).toString() << '\n'
            << "constCastNode       : " << constCastNode(NULL, t1, var).toString() << '\n'
            << "staticCastNode      : " << staticCastNode(NULL, t1, var).toString() << '\n'
            << "reinterpretCastNode : " << reinterpretCastNode(NULL, t1, var).toString() << '\n'
            << "dynamicCastNode     : " << dynamicCastNode(NULL, t1, var).toString() << '\n'
            << "parenthesesNode     : " << parenthesesNode(NULL, var).toString() << '\n'
            << "cudaCallNode        : " << cudaCallNode(NULL, var, one, two).toString() << '\n';
}

exprNode* makeExpression(const std::string &s) {
  tokenVector tokens = tokenizer::tokenize(s);
  return getExpression(tokens);
}

bool canEvaluate(const std::string &s) {
  exprNode *expr = makeExpression(s);
  bool ret = expr->canEvaluate();
  delete expr;
  return ret;
}

primitive eval(const std::string &s) {
  exprNode *expr = makeExpression(s);
  primitive value = expr->evaluate();
  delete expr;
  return value;
}

void testPairMatching() {
  exprNode *expr = makeExpression("func(0,1,2,3,4)");
  OCCA_ASSERT_EQUAL(exprNodeType::call, expr->type());
  callNode &func = expr->to<callNode>();

  OCCA_ASSERT_EQUAL("func", func.value.to<identifierNode>().value);
  OCCA_ASSERT_EQUAL(5, (int) func.args.size());
  for (int i = 0; i < 5; ++i) {
    primitiveNode &arg = func.args[i]->to<primitiveNode>();
    OCCA_ASSERT_EQUAL(i, (int) arg.value);
  }

  expr = makeExpression("(0,1,2,3,4)");
  OCCA_ASSERT_EQUAL(exprNodeType::parentheses, expr->type());

  expr = makeExpression("{0,1,2,3,4}");
  OCCA_ASSERT_EQUAL(exprNodeType::tuple, expr->type());

  expr = makeExpression("func[0 + 1]");
  OCCA_ASSERT_EQUAL(exprNodeType::subscript, expr->type());

  expr = makeExpression("func<<<0,1>>>");
  OCCA_ASSERT_EQUAL(exprNodeType::cudaCall, expr->type());
}

void testCanEvaluate() {
  OCCA_ASSERT_TRUE(canEvaluate("1 + 2 / (3)"));
  OCCA_ASSERT_FALSE(canEvaluate("1 + 2 / (3) + '1'"));
  OCCA_ASSERT_FALSE(canEvaluate("&1"));
  OCCA_ASSERT_FALSE(canEvaluate("*1"));
  OCCA_ASSERT_FALSE(canEvaluate("1::2"));
  OCCA_ASSERT_FALSE(canEvaluate("(1).(2)"));
  OCCA_ASSERT_FALSE(canEvaluate("(1).*(2)"));
  OCCA_ASSERT_FALSE(canEvaluate("1->2"));
  OCCA_ASSERT_FALSE(canEvaluate("1->*2"));
}

void testEval() {
  OCCA_ASSERT_EQUAL((int) (1 + 2 / (3)),
                    (int) eval("1 + 2 / (3)"));

  OCCA_ASSERT_EQUAL((double) ((1 + 2 / 3.1 * 4.4) / 1.2),
                    (double) eval("(1 + 2 / 3.1 * 4.4) / 1.2"));

  OCCA_ASSERT_EQUAL((int) 3,
                    (int) eval("++++1"));

  OCCA_ASSERT_EQUAL((int) 1,
                    (int) eval("1++++"));

  OCCA_ASSERT_EQUAL((int) 4,
                    (int) eval("1 ++ + ++ 2"));

  OCCA_ASSERT_EQUAL((int) 5,
                    (int) eval("1 ++ + ++ + ++ 2"));

  OCCA_ASSERT_EQUAL((int) -1,
                    (int) eval("----1"));

  OCCA_ASSERT_EQUAL((int) 1,
                    (int) eval("1----"));

  OCCA_ASSERT_EQUAL((int) 2,
                    (int) eval("1 -- + -- 2"));

  OCCA_ASSERT_EQUAL((int) 1,
                    (int) eval("1 -- + -- + -- 2"));

  OCCA_ASSERT_EQUAL((int) 1,
                    (int) eval("+ + + + + + 1"));

  OCCA_ASSERT_EQUAL((int) -1,
                    (int) eval("- - - - - 1"));

  OCCA_ASSERT_EQUAL((int) 1,
                    (int) eval("- - - - - - 1"));
}
