#include <occa/internal/utils/env.hpp>
#include <occa/internal/utils/testing.hpp>

#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/tokenizer.hpp>
#include <occa/internal/lang/builtins/types.hpp>
#include <occa/internal/lang/variable.hpp>

using namespace occa;
using namespace occa::lang;

void testPrint();
void testTernary();
void testPairMatching();
void testSpecialOperators();
void testCanEvaluate();
void testEval();

exprNode* parseExpression(const std::string &s) {
  tokenVector tokens = tokenizer_t::tokenize(s);
  return expressionParser::parse(tokens);
}

bool canEvaluate(const std::string &s) {
  exprNode *expr = parseExpression(s);
  bool ret = expr->canEvaluate();
  delete expr;
  return ret;
}

primitive eval(const std::string &s) {
  exprNode *expr = parseExpression(s);
  primitive value = expr->evaluate();
  delete expr;
  return value;
}

int main(const int argc, const char **argv) {
  testPrint();
  testTernary();
  testPairMatching();
  testSpecialOperators();
  testCanEvaluate();
  testEval();

  return 0;
}

void testStreamPrint();
void testPoutPrint();
void testDebugPrint();

void testPrint() {
  testStreamPrint();
  testPoutPrint();
  testDebugPrint();
}

void testStreamPrint() {
  primitiveNode a(NULL, 1);
  primitiveNode b(NULL, 2.0);
  primitiveNode c(NULL, false);

  qualifiers_t q1;
  q1 += volatile_;

  vartype_t t1_0(float_);
  t1_0 += const_;
  vartype_t t1_1(t1_0);
  t1_1 += pointer_t();
  vartype_t t1(t1_1);
  t1 += pointer_t();

  identifierToken varName(fileOrigin(),
                          "var");
  variable_t var_(t1, &varName);
  variableNode var(NULL, var_);

  primitiveNode one(NULL, 1);
  primitiveNode two(NULL, 2);
  primitiveNode three(NULL, 3);

  exprNodeVector args;
  args.push_back(&one);
  args.push_back(&two);
  args.push_back(&three);

  lambda_t lam;

  std::cerr
    << "\n---[ Testing << Printing ]------------------------\n"
    //---[ Left Unary ]---------------
    << "not_                : " << leftUnaryOpNode(NULL, op::not_, a).toString() << '\n'
    << "positive            : " << leftUnaryOpNode(NULL, op::positive, a).toString() << '\n'
    << "negative            : " << leftUnaryOpNode(NULL, op::negative, a).toString() << '\n'
    << "tilde               : " << leftUnaryOpNode(NULL, op::tilde, a).toString() << '\n'
    << "leftIncrement       : " << leftUnaryOpNode(NULL, op::leftIncrement, a).toString() << '\n'
    << "leftDecrement       : " << leftUnaryOpNode(NULL, op::leftDecrement, a).toString() << '\n'
    //================================

    //---[ Right Unary ]--------------
    << "rightIncrement      : " << rightUnaryOpNode(NULL, op::rightIncrement, a).toString() << '\n'
    << "rightDecrement      : " << rightUnaryOpNode(NULL, op::rightDecrement, a).toString() << '\n'
    //================================

    //---[ Binary ]-------------------
    << "add                 : " << binaryOpNode(NULL, op::add, a, b).toString() << '\n'
    << "sub                 : " << binaryOpNode(NULL, op::sub, a, b).toString() << '\n'
    << "mult                : " << binaryOpNode(NULL, op::mult, a, b).toString() << '\n'
    << "div                 : " << binaryOpNode(NULL, op::div, a, b).toString() << '\n'
    << "mod                 : " << binaryOpNode(NULL, op::mod, a, b).toString() << '\n'
    << "lessThan            : " << binaryOpNode(NULL, op::lessThan, a, b).toString() << '\n'
    << "lessThanEq          : " << binaryOpNode(NULL, op::lessThanEq, a, b).toString() << '\n'
    << "equal               : " << binaryOpNode(NULL, op::equal, a, b).toString() << '\n'
    << "compare             : " << binaryOpNode(NULL, op::compare, a, b).toString() << '\n'
    << "notEqual            : " << binaryOpNode(NULL, op::notEqual, a, b).toString() << '\n'
    << "greaterThan         : " << binaryOpNode(NULL, op::greaterThan, a, b).toString() << '\n'
    << "greaterThanEq       : " << binaryOpNode(NULL, op::greaterThanEq, a, b).toString() << '\n'
    << "and_                : " << binaryOpNode(NULL, op::and_, a, b).toString() << '\n'
    << "or_                 : " << binaryOpNode(NULL, op::or_, a, b).toString() << '\n'
    << "bitAnd              : " << binaryOpNode(NULL, op::bitAnd, a, b).toString() << '\n'
    << "bitOr               : " << binaryOpNode(NULL, op::bitOr, a, b).toString() << '\n'
    << "xor_                : " << binaryOpNode(NULL, op::xor_, a, b).toString() << '\n'
    << "leftShift           : " << binaryOpNode(NULL, op::leftShift, a, b).toString() << '\n'
    << "rightShift          : " << binaryOpNode(NULL, op::rightShift, a, b).toString() << '\n'
    << "addEq               : " << binaryOpNode(NULL, op::addEq, a, b).toString() << '\n'
    << "subEq               : " << binaryOpNode(NULL, op::subEq, a, b).toString() << '\n'
    << "multEq              : " << binaryOpNode(NULL, op::multEq, a, b).toString() << '\n'
    << "divEq               : " << binaryOpNode(NULL, op::divEq, a, b).toString() << '\n'
    << "modEq               : " << binaryOpNode(NULL, op::modEq, a, b).toString() << '\n'
    << "andEq               : " << binaryOpNode(NULL, op::andEq, a, b).toString() << '\n'
    << "orEq                : " << binaryOpNode(NULL, op::orEq, a, b).toString() << '\n'
    << "xorEq               : " << binaryOpNode(NULL, op::xorEq, a, b).toString() << '\n'
    << "leftShiftEq         : " << binaryOpNode(NULL, op::leftShiftEq, a, b).toString() << '\n'
    << "rightShiftEq        : " << binaryOpNode(NULL, op::rightShiftEq, a, b).toString() << '\n'
    //================================

//---[ Binary ]-------------------
    << "add                 : " << binaryOpNode(op::add, a, b).toString() << '\n'
    << "sub                 : " << binaryOpNode(op::sub, a, b).toString() << '\n'
    << "mult                : " << binaryOpNode(op::mult, a, b).toString() << '\n'
    << "div                 : " << binaryOpNode(op::div, a, b).toString() << '\n'
    << "mod                 : " << binaryOpNode(op::mod, a, b).toString() << '\n'
    << "lessThan            : " << binaryOpNode(op::lessThan, a, b).toString() << '\n'
    << "lessThanEq          : " << binaryOpNode(op::lessThanEq, a, b).toString() << '\n'
    << "equal               : " << binaryOpNode(op::equal, a, b).toString() << '\n'
    << "compare             : " << binaryOpNode(op::compare, a, b).toString() << '\n'
    << "notEqual            : " << binaryOpNode(op::notEqual, a, b).toString() << '\n'
    << "greaterThan         : " << binaryOpNode(op::greaterThan, a, b).toString() << '\n'
    << "greaterThanEq       : " << binaryOpNode(op::greaterThanEq, a, b).toString() << '\n'
    << "and_                : " << binaryOpNode(op::and_, a, b).toString() << '\n'
    << "or_                 : " << binaryOpNode(op::or_, a, b).toString() << '\n'
    << "bitAnd              : " << binaryOpNode(op::bitAnd, a, b).toString() << '\n'
    << "bitOr               : " << binaryOpNode(op::bitOr, a, b).toString() << '\n'
    << "xor_                : " << binaryOpNode(op::xor_, a, b).toString() << '\n'
    << "leftShift           : " << binaryOpNode(op::leftShift, a, b).toString() << '\n'
    << "rightShift          : " << binaryOpNode(op::rightShift, a, b).toString() << '\n'
    << "addEq               : " << binaryOpNode(op::addEq, a, b).toString() << '\n'
    << "subEq               : " << binaryOpNode(op::subEq, a, b).toString() << '\n'
    << "multEq              : " << binaryOpNode(op::multEq, a, b).toString() << '\n'
    << "divEq               : " << binaryOpNode(op::divEq, a, b).toString() << '\n'
    << "modEq               : " << binaryOpNode(op::modEq, a, b).toString() << '\n'
    << "andEq               : " << binaryOpNode(op::andEq, a, b).toString() << '\n'
    << "orEq                : " << binaryOpNode(op::orEq, a, b).toString() << '\n'
    << "xorEq               : " << binaryOpNode(op::xorEq, a, b).toString() << '\n'
    << "leftShiftEq         : " << binaryOpNode(op::leftShiftEq, a, b).toString() << '\n'
    << "rightShiftEq        : " << binaryOpNode(op::rightShiftEq, a, b).toString() << '\n'
    //================================

    //---[ Ternary ]------------------
    << "ternary             : " << ternaryOpNode(a, b, c).toString() << '\n'
    //================================

    //---[ Other Nodes ]--------------
    << "one                 : " << one.toString() << '\n'
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
    << "cudaCallNode        : " << cudaCallNode(NULL, var, one, two).toString() << '\n'
    << "dpcppAtomicNode     : " << dpcppAtomicNode(NULL,t1,var).toString() << '\n'
    << "dpcppLocalMemoryNode: " << dpcppLocalMemoryNode(NULL,t1,"cgh").toString() << '\n'
    << "lambdaNode          : " << lambdaNode(NULL,lam).toString() << '\n'
    //================================
    ;
}

void testPoutPrint() {
  primitiveNode a(NULL, 1);
  primitiveNode b(NULL, 2.0);
  primitiveNode c(NULL, false);

  qualifiers_t q1;
  q1 += volatile_;

  vartype_t t1_0(float_);
  t1_0 += const_;
  vartype_t t1_1(t1_0);
  t1_1 += pointer_t();
  vartype_t t1(t1_1);
  t1 += pointer_t();

  identifierToken varName(fileOrigin(),
                          "var");
  variable_t var_(t1, &varName);
  variableNode var(NULL, var_);

  primitiveNode one(NULL, 1);
  primitiveNode two(NULL, 2);
  primitiveNode three(NULL, 3);

  exprNodeVector args;
  args.push_back(&one);
  args.push_back(&two);
  args.push_back(&three);

  lambda_t lam;

  printer pout(io::stderr);

  std::cerr << "\n---[ Testing pout Printing ]----------------------\n";

  //---[ Left Unary ]---------------
  leftUnaryOpNode(NULL, op::not_, a).print(pout); pout << '\n';
  leftUnaryOpNode(NULL, op::positive, a).print(pout); pout << '\n';
  leftUnaryOpNode(NULL, op::negative, a).print(pout); pout << '\n';
  leftUnaryOpNode(NULL, op::tilde, a).print(pout); pout << '\n';
  leftUnaryOpNode(NULL, op::leftIncrement, a).print(pout); pout << '\n';
  leftUnaryOpNode(NULL, op::leftDecrement, a).print(pout); pout << '\n';
  //================================

  //---[ Right Unary ]--------------
  rightUnaryOpNode(NULL, op::rightIncrement, a).print(pout); pout << '\n';
  rightUnaryOpNode(NULL, op::rightDecrement, a).print(pout); pout << '\n';
  //================================

  //---[ Binary ]-------------------
  binaryOpNode(NULL, op::add, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::sub, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::mult, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::div, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::mod, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::lessThan, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::lessThanEq, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::equal, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::compare, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::notEqual, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::greaterThan, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::greaterThanEq, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::and_, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::or_, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::bitAnd, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::bitOr, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::xor_, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::leftShift, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::rightShift, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::addEq, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::subEq, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::multEq, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::divEq, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::modEq, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::andEq, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::orEq, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::xorEq, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::leftShiftEq, a, b).print(pout); pout << '\n';
  binaryOpNode(NULL, op::rightShiftEq, a, b).print(pout); pout << '\n';
  //================================

//---[ Binary ]-------------------
  binaryOpNode(op::add, a, b).print(pout); pout << '\n';
  binaryOpNode(op::sub, a, b).print(pout); pout << '\n';
  binaryOpNode(op::mult, a, b).print(pout); pout << '\n';
  binaryOpNode(op::div, a, b).print(pout); pout << '\n';
  binaryOpNode(op::mod, a, b).print(pout); pout << '\n';
  binaryOpNode(op::lessThan, a, b).print(pout); pout << '\n';
  binaryOpNode(op::lessThanEq, a, b).print(pout); pout << '\n';
  binaryOpNode(op::equal, a, b).print(pout); pout << '\n';
  binaryOpNode(op::compare, a, b).print(pout); pout << '\n';
  binaryOpNode(op::notEqual, a, b).print(pout); pout << '\n';
  binaryOpNode(op::greaterThan, a, b).print(pout); pout << '\n';
  binaryOpNode(op::greaterThanEq, a, b).print(pout); pout << '\n';
  binaryOpNode(op::and_, a, b).print(pout); pout << '\n';
  binaryOpNode(op::or_, a, b).print(pout); pout << '\n';
  binaryOpNode(op::bitAnd, a, b).print(pout); pout << '\n';
  binaryOpNode(op::bitOr, a, b).print(pout); pout << '\n';
  binaryOpNode(op::xor_, a, b).print(pout); pout << '\n';
  binaryOpNode(op::leftShift, a, b).print(pout); pout << '\n';
  binaryOpNode(op::rightShift, a, b).print(pout); pout << '\n';
  binaryOpNode(op::addEq, a, b).print(pout); pout << '\n';
  binaryOpNode(op::subEq, a, b).print(pout); pout << '\n';
  binaryOpNode(op::multEq, a, b).print(pout); pout << '\n';
  binaryOpNode(op::divEq, a, b).print(pout); pout << '\n';
  binaryOpNode(op::modEq, a, b).print(pout); pout << '\n';
  binaryOpNode(op::andEq, a, b).print(pout); pout << '\n';
  binaryOpNode(op::orEq, a, b).print(pout); pout << '\n';
  binaryOpNode(op::xorEq, a, b).print(pout); pout << '\n';
  binaryOpNode(op::leftShiftEq, a, b).print(pout); pout << '\n';
  binaryOpNode(op::rightShiftEq, a, b).print(pout); pout << '\n';
  //================================

  //---[ Ternary ]------------------
  ternaryOpNode(a, b, c).print(pout); pout << '\n';
  //================================

  //---[ Other Nodes ]--------------
  one.print(pout); pout << '\n';
  var.print(pout); pout << '\n';
  subscriptNode(NULL, var, one).print(pout); pout << '\n';
  callNode(NULL, var, args).print(pout); pout << '\n';
  newNode(NULL, t1, var, three).print(pout); pout << '\n';
  newNode(NULL, t1, var).print(pout); pout << '\n';
  deleteNode(NULL, var, false).print(pout); pout << '\n';
  deleteNode(NULL, var, true).print(pout); pout << '\n';
  throwNode(NULL, one).print(pout); pout << '\n';
  sizeofNode(NULL, var).print(pout); pout << '\n';
  funcCastNode(NULL, t1, var).print(pout); pout << '\n';
  parenCastNode(NULL, t1, var).print(pout); pout << '\n';
  constCastNode(NULL, t1, var).print(pout); pout << '\n';
  staticCastNode(NULL, t1, var).print(pout); pout << '\n';
  reinterpretCastNode(NULL, t1, var).print(pout); pout << '\n';
  dynamicCastNode(NULL, t1, var).print(pout); pout << '\n';
  parenthesesNode(NULL, var).print(pout); pout << '\n';
  cudaCallNode(NULL, var, one, two).print(pout); pout << '\n';
  dpcppAtomicNode(NULL,t1,var).print(pout); pout << '\n';
  dpcppLocalMemoryNode(NULL,t1,"cgh").print(pout); pout << '\n';
  lambdaNode(NULL, lam).print(pout); pout << '\n';
  //================================
}

void testDebugPrint() {
  primitiveNode a(NULL, 1);
  primitiveNode b(NULL, 2.0);
  primitiveNode c(NULL, false);

  qualifiers_t q1;
  q1 += volatile_;

  vartype_t t1_0(float_);
  t1_0 += const_;
  vartype_t t1_1(t1_0);
  t1_1 += pointer_t();
  vartype_t t1(t1_1);
  t1 += pointer_t();

  identifierToken varName(fileOrigin(),
                          "var");
  variable_t var_(t1, &varName);
  variableNode var(NULL, var_);

  primitiveNode one(NULL, 1);
  primitiveNode two(NULL, 2);
  primitiveNode three(NULL, 3);

  exprNodeVector args;
  args.push_back(&one);
  args.push_back(&two);
  args.push_back(&three);

  lambda_t lam;


  std::cerr << "\n---[ Testing Debug Printing ]---------------------\n";

  //---[ Left Unary ]---------------
  leftUnaryOpNode(NULL, op::not_, a).debugPrint("");
  leftUnaryOpNode(NULL, op::positive, a).debugPrint("");
  leftUnaryOpNode(NULL, op::negative, a).debugPrint("");
  leftUnaryOpNode(NULL, op::tilde, a).debugPrint("");
  leftUnaryOpNode(NULL, op::leftIncrement, a).debugPrint("");
  leftUnaryOpNode(NULL, op::leftDecrement, a).debugPrint("");
  //================================

  //---[ Right Unary ]--------------
  rightUnaryOpNode(NULL, op::rightIncrement, a).debugPrint("");
  rightUnaryOpNode(NULL, op::rightDecrement, a).debugPrint("");
  //================================

  //---[ Binary ]-------------------
  binaryOpNode(NULL, op::add, a, b).debugPrint("");
  binaryOpNode(NULL, op::sub, a, b).debugPrint("");
  binaryOpNode(NULL, op::mult, a, b).debugPrint("");
  binaryOpNode(NULL, op::div, a, b).debugPrint("");
  binaryOpNode(NULL, op::mod, a, b).debugPrint("");
  binaryOpNode(NULL, op::lessThan, a, b).debugPrint("");
  binaryOpNode(NULL, op::lessThanEq, a, b).debugPrint("");
  binaryOpNode(NULL, op::equal, a, b).debugPrint("");
  binaryOpNode(NULL, op::compare, a, b).debugPrint("");
  binaryOpNode(NULL, op::notEqual, a, b).debugPrint("");
  binaryOpNode(NULL, op::greaterThan, a, b).debugPrint("");
  binaryOpNode(NULL, op::greaterThanEq, a, b).debugPrint("");
  binaryOpNode(NULL, op::and_, a, b).debugPrint("");
  binaryOpNode(NULL, op::or_, a, b).debugPrint("");
  binaryOpNode(NULL, op::bitAnd, a, b).debugPrint("");
  binaryOpNode(NULL, op::bitOr, a, b).debugPrint("");
  binaryOpNode(NULL, op::xor_, a, b).debugPrint("");
  binaryOpNode(NULL, op::leftShift, a, b).debugPrint("");
  binaryOpNode(NULL, op::rightShift, a, b).debugPrint("");
  binaryOpNode(NULL, op::addEq, a, b).debugPrint("");
  binaryOpNode(NULL, op::subEq, a, b).debugPrint("");
  binaryOpNode(NULL, op::multEq, a, b).debugPrint("");
  binaryOpNode(NULL, op::divEq, a, b).debugPrint("");
  binaryOpNode(NULL, op::modEq, a, b).debugPrint("");
  binaryOpNode(NULL, op::andEq, a, b).debugPrint("");
  binaryOpNode(NULL, op::orEq, a, b).debugPrint("");
  binaryOpNode(NULL, op::xorEq, a, b).debugPrint("");
  binaryOpNode(NULL, op::leftShiftEq, a, b).debugPrint("");
  binaryOpNode(NULL, op::rightShiftEq, a, b).debugPrint("");
  //================================

  //---[ Ternary ]------------------
  ternaryOpNode(a, b, c).debugPrint("");
  //================================

  //---[ Other Nodes ]--------------
  one.debugPrint("");
  var.debugPrint("");
  subscriptNode(NULL, var, one).debugPrint("");
  callNode(NULL, var, args).debugPrint("");
  newNode(NULL, t1, var, three).debugPrint("");
  newNode(NULL, t1, var).debugPrint("");
  deleteNode(NULL, var, false).debugPrint("");
  deleteNode(NULL, var, true).debugPrint("");
  throwNode(NULL, one).debugPrint("");
  sizeofNode(NULL, var).debugPrint("");
  funcCastNode(NULL, t1, var).debugPrint("");
  parenCastNode(NULL, t1, var).debugPrint("");
  constCastNode(NULL, t1, var).debugPrint("");
  staticCastNode(NULL, t1, var).debugPrint("");
  reinterpretCastNode(NULL, t1, var).debugPrint("");
  dynamicCastNode(NULL, t1, var).debugPrint("");
  parenthesesNode(NULL, var).debugPrint("");
  cudaCallNode(NULL, var, one, two).debugPrint("");
  dpcppAtomicNode(NULL,t1,var).debugPrint("");
  dpcppLocalMemoryNode(NULL,t1,"cgh").debugPrint("");
  lambdaNode(NULL, lam).debugPrint("");
  //================================
}

void testTernary() {
  exprNode *expr;

  expr = parseExpression("a = true ? 1 : 2");
  expr->debugPrint();
  delete expr;

  expr = parseExpression("a = true ? (false ? 1 : (false ? 2 : 3)) : 4");
  expr->debugPrint();
  delete expr;
}

void testPairMatching() {
  exprNode *expr = parseExpression("func(0,1,2,3,4)");
  ASSERT_EQ_BINARY(exprNodeType::call,
                   expr->type());
  callNode &func = expr->to<callNode>();

  ASSERT_EQ("func", func.value->to<identifierNode>().value);
  ASSERT_EQ(5, (int) func.args.size());
  for (int i = 0; i < 5; ++i) {
    primitiveNode &arg = func.args[i]->to<primitiveNode>();
    ASSERT_EQ(i, (int) arg.value);
  }

  delete expr;
  expr = parseExpression("(0,1,2,3,4)");
  ASSERT_EQ_BINARY(exprNodeType::parentheses,
                   expr->type());

  delete expr;
  expr = parseExpression("{0,1,2,3,4}");
  ASSERT_EQ_BINARY(exprNodeType::tuple,
                   expr->type());

  delete expr;
  expr = parseExpression("array[0 + 1]");
  ASSERT_EQ_BINARY(exprNodeType::subscript,
                   expr->type());

  delete expr;
  expr = parseExpression("func<<<0,1>>>");
  ASSERT_EQ_BINARY(exprNodeType::cudaCall,
                   expr->type());

  delete expr;

  std::cerr << "\nTesting pair errors:\n";
  parseExpression("(0,1,2]");
  parseExpression("[0,1,2}");
  parseExpression("{0,1,2)");
  parseExpression("<<<0,1,2)");
}

void testSpecialOperators() {
  exprNode *expr = parseExpression("sizeof(int)");
  ASSERT_EQ_BINARY(exprNodeType::sizeof_,
                   expr->type());

  delete expr;
  expr = parseExpression("throw 2 + 2");
  ASSERT_EQ_BINARY(exprNodeType::throw_,
                   expr->type());

  delete expr;

  std::cerr << "\nTesting unsupported new and delete:\n";
  expr = parseExpression("new int[2]");
  expr = parseExpression("delete foo");
  expr = parseExpression("delete [] foo");
}

void testCanEvaluate() {
  ASSERT_TRUE(canEvaluate("1 + 2 / (3)"));
  ASSERT_FALSE(canEvaluate("1 + 2 / (3) + '1'"));
  ASSERT_FALSE(canEvaluate("&1"));
  ASSERT_FALSE(canEvaluate("*1"));
  ASSERT_FALSE(canEvaluate("1::2"));
  ASSERT_FALSE(canEvaluate("(1).(2)"));
  ASSERT_FALSE(canEvaluate("(1).*(2)"));
  ASSERT_FALSE(canEvaluate("1->2"));
  ASSERT_FALSE(canEvaluate("1->*2"));
}

void testEval() {
  ASSERT_EQ((int) (1 + (2 * 3)),
            (int) eval("1 + (2 * 3)"));

  ASSERT_EQ((int) (1 + 2 / (3)),
            (int) eval("1 + 2 / (3)"));

  ASSERT_EQ((double) ((1 + 2 / 3.1 * 4.4) / 1.2),
            (double) eval("(1 + 2 / 3.1 * 4.4) / 1.2"));

  ASSERT_EQ((int) 3,
            (int) eval("++++1"));

  ASSERT_EQ((int) 1,
            (int) eval("1++++"));

  ASSERT_EQ((int) 4,
            (int) eval("1 ++ + ++ 2"));

  ASSERT_EQ((int) 5,
            (int) eval("1 ++ + ++ + ++ 2"));

  ASSERT_EQ((int) -1,
            (int) eval("----1"));

  ASSERT_EQ((int) 1,
            (int) eval("1----"));

  ASSERT_EQ((int) 2,
            (int) eval("1 -- + -- 2"));

  ASSERT_EQ((int) 1,
            (int) eval("1 -- + -- + -- 2"));

  ASSERT_EQ((int) 1,
            (int) eval("+ + + + + + 1"));

  ASSERT_EQ((int) -1,
            (int) eval("- - - - - 1"));

  ASSERT_EQ((int) 1,
            (int) eval("- - - - - - 1"));
}
