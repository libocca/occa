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

void testOperatorNodes();
void testOtherNodes();

int main(const int argc, const char **argv) {
  testOperatorNodes();
  testOtherNodes();
}

using namespace occa::lang;

void testOperatorNodes() {
  primitiveNode a(1);
  primitiveNode b(2.0);
  primitiveNode c(false);

  std::cout
    //---[ Left Unary ]---------------
    << "not_           : " << leftUnaryOpNode(op::not_, a).toString() << '\n'
    << "positive       : " << leftUnaryOpNode(op::positive, a).toString() << '\n'
    << "negative       : " << leftUnaryOpNode(op::negative, a).toString() << '\n'
    << "tilde          : " << leftUnaryOpNode(op::tilde, a).toString() << '\n'
    << "leftIncrement  : " << leftUnaryOpNode(op::leftIncrement, a).toString() << '\n'
    << "leftDecrement  : " << leftUnaryOpNode(op::leftDecrement, a).toString() << '\n'
    //================================

    //---[ Right Unary ]--------------
    << "rightIncrement : " << rightUnaryOpNode(op::rightIncrement, a).toString() << '\n'
    << "rightDecrement : " << rightUnaryOpNode(op::rightDecrement, a).toString() << '\n'
    //================================

    //---[ Binary ]-------------------
    << "add            : " << binaryOpNode(op::add, a, b).toString() << '\n'
    << "sub            : " << binaryOpNode(op::sub, a, b).toString() << '\n'
    << "mult           : " << binaryOpNode(op::mult, a, b).toString() << '\n'
    << "div            : " << binaryOpNode(op::div, a, b).toString() << '\n'
    << "mod            : " << binaryOpNode(op::mod, a, b).toString() << '\n'
    << "lessThan       : " << binaryOpNode(op::lessThan, a, b).toString() << '\n'
    << "lessThanEq     : " << binaryOpNode(op::lessThanEq, a, b).toString() << '\n'
    << "equal          : " << binaryOpNode(op::equal, a, b).toString() << '\n'
    << "notEqual       : " << binaryOpNode(op::notEqual, a, b).toString() << '\n'
    << "greaterThan    : " << binaryOpNode(op::greaterThan, a, b).toString() << '\n'
    << "greaterThanEq  : " << binaryOpNode(op::greaterThanEq, a, b).toString() << '\n'
    << "and_           : " << binaryOpNode(op::and_, a, b).toString() << '\n'
    << "or_            : " << binaryOpNode(op::or_, a, b).toString() << '\n'
    << "bitAnd         : " << binaryOpNode(op::bitAnd, a, b).toString() << '\n'
    << "bitOr          : " << binaryOpNode(op::bitOr, a, b).toString() << '\n'
    << "xor_           : " << binaryOpNode(op::xor_, a, b).toString() << '\n'
    << "leftShift      : " << binaryOpNode(op::leftShift, a, b).toString() << '\n'
    << "rightShift     : " << binaryOpNode(op::rightShift, a, b).toString() << '\n'
    << "addEq          : " << binaryOpNode(op::addEq, a, b).toString() << '\n'
    << "subEq          : " << binaryOpNode(op::subEq, a, b).toString() << '\n'
    << "multEq         : " << binaryOpNode(op::multEq, a, b).toString() << '\n'
    << "divEq          : " << binaryOpNode(op::divEq, a, b).toString() << '\n'
    << "modEq          : " << binaryOpNode(op::modEq, a, b).toString() << '\n'
    << "andEq          : " << binaryOpNode(op::andEq, a, b).toString() << '\n'
    << "orEq           : " << binaryOpNode(op::orEq, a, b).toString() << '\n'
    << "xorEq          : " << binaryOpNode(op::xorEq, a, b).toString() << '\n'
    << "leftShiftEq    : " << binaryOpNode(op::leftShiftEq, a, b).toString() << '\n'
    << "rightShiftEq   : " << binaryOpNode(op::rightShiftEq, a, b).toString() << '\n'
    //================================

    //---[ Ternary ]------------------
    << "ternary        : " << ternaryOpNode(op::ternary, a, b, c).toString() << '\n'
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
  variableNode var(var_);

  primitiveNode one(1), two(2), three(3);
  exprNodeVector_t args;
  args.push_back(&one);
  args.push_back(&two);
  args.push_back(&three);

  std::cout << "one                 : " << one.toString() << '\n'
            << "var                 : " << var.toString() << '\n'
            << "subscript           : " << subscriptNode(var, one).toString() << '\n'
            << "callNode            : " << callNode(var, args).toString() << '\n'
            << "newNode             : " << newNode(t1, var, three).toString() << '\n'
            << "newNode             : " << newNode(t1, var).toString() << '\n'
            << "deleteNode          : " << deleteNode(var, false).toString() << '\n'
            << "deleteNode          : " << deleteNode(var, true).toString() << '\n'
            << "throwNode           : " << throwNode(one).toString() << '\n'
            << "sizeofNode          : " << sizeofNode(var).toString() << '\n'
            << "funcCastNode        : " << funcCastNode(t1, var).toString() << '\n'
            << "parenCastNode       : " << parenCastNode(t1, var).toString() << '\n'
            << "constCastNode       : " << constCastNode(t1, var).toString() << '\n'
            << "staticCastNode      : " << staticCastNode(t1, var).toString() << '\n'
            << "reinterpretCastNode : " << reinterpretCastNode(t1, var).toString() << '\n'
            << "dynamicCastNode     : " << dynamicCastNode(t1, var).toString() << '\n'
            << "parenthesesNode     : " << parenthesesNode(var).toString() << '\n'
            << "cudaCallNode        : " << cudaCallNode(one, two).toString() << '\n';
}
