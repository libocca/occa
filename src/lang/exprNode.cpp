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
#include <occa/lang/expression.hpp>
#include <occa/lang/variable.hpp>
#include <occa/tools/string.hpp>

namespace occa {
  namespace lang {
    namespace exprNodeType {
      const udim_t empty             = (1L << 0);
      const udim_t primitive         = (1L << 1);
      const udim_t char_             = (1L << 2);
      const udim_t string            = (1L << 3);
      const udim_t identifier        = (1L << 4);
      const udim_t type              = (1L << 5);
      const udim_t vartype           = (1L << 6);
      const udim_t variable          = (1L << 7);
      const udim_t function          = (1L << 8);

      const udim_t value             = (primitive |
                                        type      |
                                        vartype   |
                                        variable  |
                                        function);

      const udim_t rawOp             = (1L << 9);
      const udim_t leftUnary         = (1L << 10);
      const udim_t rightUnary        = (1L << 11);
      const udim_t binary            = (1L << 12);
      const udim_t ternary           = (1L << 13);
      const udim_t op                = (leftUnary  |
                                        rightUnary |
                                        binary     |
                                        ternary);

      const udim_t pair              = (1L << 14);

      const udim_t subscript         = (1L << 15);
      const udim_t call              = (1L << 16);

      const udim_t sizeof_           = (1L << 17);
      const udim_t sizeof_pack_      = (1L << 18);
      const udim_t new_              = (1L << 19);
      const udim_t delete_           = (1L << 20);
      const udim_t throw_            = (1L << 21);

      const udim_t typeid_           = (1L << 22);
      const udim_t noexcept_         = (1L << 23);
      const udim_t alignof_          = (1L << 24);

      const udim_t const_cast_       = (1L << 25);
      const udim_t dynamic_cast_     = (1L << 26);
      const udim_t static_cast_      = (1L << 27);
      const udim_t reinterpret_cast_ = (1L << 28);

      const udim_t funcCast          = (1L << 29);
      const udim_t parenCast         = (1L << 30);
      const udim_t constCast         = (1L << 31);
      const udim_t staticCast        = (1L << 32);
      const udim_t reinterpretCast   = (1L << 33);
      const udim_t dynamicCast       = (1L << 34);

      const udim_t parentheses       = (1L << 35);
      const udim_t tuple             = (1L << 36);
      const udim_t cudaCall          = (1L << 37);
    }

    exprNode::exprNode(token_t *token_) :
      token(token_t::clone(token_)) {}

    exprNode::~exprNode() {
      delete token;
    }

    exprNode* exprNode::clone(exprNode *expr) {
      if (!expr) {
        return NULL;
      }
      return expr->clone();
    }

    bool exprNode::canEvaluate() const {
      return false;
    }

    primitive exprNode::evaluate() const {
      return primitive();
    }

    exprNode* exprNode::startNode() {
      return this;
    }

    exprNode* exprNode::endNode() {
      return this;
    }

    bool exprNode::hasAttribute(const std::string &attr) const {
      return false;
    }

    variable_t* exprNode::getVariable() {
      return NULL;
    }

    exprNode* exprNode::wrapInParentheses() {
      return clone();
    }

    std::string exprNode::toString() const {
      std::stringstream ss;
      printer pout(ss);
      pout << (*this);
      return ss.str();
    }

    void exprNode::printWarning(const std::string &message) const {
      token->printWarning(message);
    }

    void exprNode::printError(const std::string &message) const {
      token->printError(message);
    }

    void exprNode::debugPrint() const {
      debugPrint("");
      std::cerr << '\n';
    }

    void exprNode::childDebugPrint(const std::string &prefix) const {
      debugPrint(prefix + "|   ");
    }

    printer& operator << (printer &pout,
                          const exprNode &node) {
      node.print(pout);
      return pout;
    }

    void cloneExprNodeVector(exprNodeVector &dest,
                             const exprNodeVector &src) {
      const int nodes = (int) src.size();
      dest.clear();
      dest.reserve(nodes);
      for (int i = 0; i < nodes; ++i) {
        dest.push_back(src[i]->clone());
      }
    }

    void freeExprNodeVector(exprNodeVector &vec) {
      const int nodes = (int) vec.size();
      for (int i = 0; i < nodes; ++i) {
        delete vec[i];
      }
      vec.clear();
    }

    //---[ Empty ]----------------------
    emptyNode::emptyNode() :
      exprNode(NULL) {}

    emptyNode::~emptyNode() {}

    udim_t emptyNode::type() const {
      return exprNodeType::empty;
    }

    exprNode* emptyNode::clone() const {
      return new emptyNode();
    }

    void emptyNode::setChildren(exprNodeRefVector &children) {}

    void emptyNode::print(printer &pout) const {}

    void emptyNode::debugPrint(const std::string &prefix) const {
      std::cerr << prefix << "|\n"
                << prefix << "|---o\n"
                << prefix << '\n';
    }

    const emptyNode noExprNode;
    //==================================

    //---[ Values ]---------------------
    //  |---[ Primitive ]---------------
    primitiveNode::primitiveNode(token_t *token_,
                                 primitive value_) :
      exprNode(token_),
      value(value_) {}

    primitiveNode::primitiveNode(const primitiveNode &node) :
      exprNode(node.token),
      value(node.value) {}

    primitiveNode::~primitiveNode() {}

    udim_t primitiveNode::type() const {
      return exprNodeType::primitive;
    }

    exprNode* primitiveNode::clone() const {
      return new primitiveNode(token, value);
    }

    bool primitiveNode::canEvaluate() const {
      return true;
    }

    primitive primitiveNode::evaluate() const {
      return value;
    }

    void primitiveNode::setChildren(exprNodeRefVector &children) {}

    void primitiveNode::print(printer &pout) const {
      pout << value.toString();
    }

    void primitiveNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << (*this);
      std::cerr << "] (primitive)\n";
    }
    //  |===============================

    //  |---[ Char ]--------------------
    charNode::charNode(token_t *token_,
                       const std::string &value_) :
      exprNode(token_),
      value(value_) {}

    charNode::charNode(const charNode &node) :
      exprNode(node.token),
      value(node.value) {}

    charNode::~charNode() {}

    udim_t charNode::type() const {
      return exprNodeType::char_;
    }

    exprNode* charNode::clone() const {
      return new charNode(token, value);
    }

    void charNode::setChildren(exprNodeRefVector &children) {}

    void charNode::print(printer &pout) const {
      pout << "'" << escape(value, '\'') << '"';
    }

    void charNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << '\n'
                << prefix << "|---[";
      pout << (*this);
      std::cerr << "] (char)\n";
    }
    //  |===============================

    //  |---[ String ]------------------
    stringNode::stringNode(token_t *token_,
                           const std::string &value_) :
      exprNode(token_),
      value(value_) {}

    stringNode::stringNode(const stringNode &node) :
      exprNode(node.token),
      value(node.value) {}

    stringNode::~stringNode() {}

    udim_t stringNode::type() const {
      return exprNodeType::string;
    }

    exprNode* stringNode::clone() const {
      return new stringNode(token, value);
    }

    void stringNode::setChildren(exprNodeRefVector &children) {}

    void stringNode::print(printer &pout) const {
      pout << "\"" << escape(value, '"') << "\"";
    }

    void stringNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << (*this);
      std::cerr << "] (string)\n";
    }
    //  |===============================

    //  |---[ Identifier ]--------------
    identifierNode::identifierNode(token_t *token_,
                                   const std::string &value_) :
      exprNode(token_),
      value(value_) {}

    identifierNode::identifierNode(const identifierNode &node) :
      exprNode(node.token),
      value(node.value) {}

    identifierNode::~identifierNode() {}

    udim_t identifierNode::type() const {
      return exprNodeType::identifier;
    }

    exprNode* identifierNode::clone() const {
      return new identifierNode(token, value);
    }

    void identifierNode::setChildren(exprNodeRefVector &children) {}

    void identifierNode::print(printer &pout) const {
      pout << value;
    }

    void identifierNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << '\n'
                << prefix << "|---[";
      pout << (*this);
      std::cerr << "] (identifier)\n";
    }
    //  |===============================

    //  |---[ Type ]--------------------
    typeNode::typeNode(token_t *token_,
                       type_t &value_) :
      exprNode(token_),
      value(value_) {}

    typeNode::typeNode(const typeNode &node) :
      exprNode(node.token),
      value(node.value) {}

    typeNode::~typeNode() {}

    udim_t typeNode::type() const {
      return exprNodeType::type;
    }

    exprNode* typeNode::clone() const {
      return new typeNode(token, value);
    }

    void typeNode::setChildren(exprNodeRefVector &children) {}

    bool typeNode::hasAttribute(const std::string &attr) const {
      return value.hasAttribute(attr);
    }

    void typeNode::print(printer &pout) const {
      pout << value;
    }

    void typeNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << (*this);
      std::cerr << "] (type)\n";
    }
    //  |===============================

    //  |---[ Vartype ]-----------------
    vartypeNode::vartypeNode(token_t *token_,
                             const vartype_t &value_) :
      exprNode(token_),
      value(value_) {}

    vartypeNode::vartypeNode(const vartypeNode &node) :
      exprNode(node.token),
      value(node.value) {}

    vartypeNode::~vartypeNode() {}

    udim_t vartypeNode::type() const {
      return exprNodeType::vartype;
    }

    exprNode* vartypeNode::clone() const {
      return new vartypeNode(token, value);
    }

    void vartypeNode::setChildren(exprNodeRefVector &children) {}

    bool vartypeNode::hasAttribute(const std::string &attr) const {
      return value.hasAttribute(attr);
    }

    void vartypeNode::print(printer &pout) const {
      pout << value;
    }

    void vartypeNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << (*this);
      std::cerr << "] (vartype)\n";
    }
    //  |===============================

    //  |---[ Variable ]----------------
    variableNode::variableNode(token_t *token_,
                               variable_t &value_) :
      exprNode(token_),
      value(value_) {}

    variableNode::variableNode(const variableNode &node) :
      exprNode(node.token),
      value(node.value) {}

    variableNode::~variableNode() {}

    udim_t variableNode::type() const {
      return exprNodeType::variable;
    }

    exprNode* variableNode::clone() const {
      return new variableNode(token, value);
    }

    void variableNode::setChildren(exprNodeRefVector &children) {}

    bool variableNode::hasAttribute(const std::string &attr) const {
      return value.hasAttribute(attr);
    }

    variable_t* variableNode::getVariable() {
      return &value;
    }

    void variableNode::print(printer &pout) const {
      pout << value;
    }

    void variableNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << (*this);
      std::cerr << "] (variable)\n";
    }
    //  |===============================

    //  |---[ Function ]----------------
    functionNode::functionNode(token_t *token_,
                               function_t &value_) :
      exprNode(token_),
      value(value_) {}

    functionNode::functionNode(const functionNode &node) :
      exprNode(node.token),
      value(node.value) {}

    functionNode::~functionNode() {}

    udim_t functionNode::type() const {
      return exprNodeType::function;
    }

    exprNode* functionNode::clone() const {
      return new functionNode(token, value);
    }

    void functionNode::setChildren(exprNodeRefVector &children) {}

    bool functionNode::hasAttribute(const std::string &attr) const {
      return value.hasAttribute(attr);
    }

    void functionNode::print(printer &pout) const {
      pout << value;
    }

    void functionNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << (*this);
      std::cerr << "] (function)\n";
    }
    //  |===============================
    //==================================

    //---[ Operators ]------------------
    exprOpNode::exprOpNode(operatorToken &token_) :
      exprNode(&token_),
      op(*(token_.op)) {}

    exprOpNode::exprOpNode(token_t *token_,
                           const operator_t &op_) :
      exprNode(token_),
      op(op_) {}

    opType_t exprOpNode::opType() const {
      return op.opType;
    }

    udim_t exprOpNode::type() const {
      return exprNodeType::rawOp;
    }

    exprNode* exprOpNode::clone() const {
      return new exprOpNode(token, op);
    }

    void exprOpNode::setChildren(exprNodeRefVector &children) {}

    exprNode* exprOpNode::wrapInParentheses() {
      return new parenthesesNode(token, *this);
    }

    void exprOpNode::print(printer &pout) const {
      token->printError("[Waldo] (exprOpNode) Unsure how you got here...");
    }

    void exprOpNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << op;
      std::cerr << "] (exprOpNode)\n";
    }

    leftUnaryOpNode::leftUnaryOpNode(token_t *token_,
                                     const unaryOperator_t &op_,
                                     const exprNode &value_) :
      exprOpNode(token_, op_),
      value(value_.clone()) {}

    leftUnaryOpNode::leftUnaryOpNode(const leftUnaryOpNode &node) :
      exprOpNode(node.token, node.op),
      value(node.value->clone()) {}

    leftUnaryOpNode::~leftUnaryOpNode() {
      delete value;
    }

    udim_t leftUnaryOpNode::type() const {
      return exprNodeType::leftUnary;
    }

    exprNode* leftUnaryOpNode::clone() const {
      return new leftUnaryOpNode(token,
                                 (const unaryOperator_t&) op,
                                 *value);
    }

    bool leftUnaryOpNode::canEvaluate() const {
      if (op.opType & (operatorType::dereference |
                       operatorType::address)) {
        return false;
      }
      return value->canEvaluate();
    }

    primitive leftUnaryOpNode::evaluate() const {
      primitive pValue = value->evaluate();
      return ((unaryOperator_t&) op)(pValue);
    }

    exprNode* leftUnaryOpNode::endNode() {
      return value->endNode();
    }

    void leftUnaryOpNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
    }

    variable_t* leftUnaryOpNode::getVariable() {
      return value->getVariable();
    }

    void leftUnaryOpNode::print(printer &pout) const {
      pout << op << *value;
    }

    void leftUnaryOpNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << op;
      std::cerr << "] (leftUnary)\n";
      value->childDebugPrint(prefix);
    }

    rightUnaryOpNode::rightUnaryOpNode(token_t *token_,
                                       const unaryOperator_t &op_,
                                       const exprNode &value_) :
      exprOpNode(token_, op_),
      value(value_.clone()) {}

    rightUnaryOpNode::rightUnaryOpNode(const rightUnaryOpNode &node) :
      exprOpNode(node.token, node.op),
      value(node.value->clone()) {}

    rightUnaryOpNode::~rightUnaryOpNode() {
      delete value;
    }

    udim_t rightUnaryOpNode::type() const {
      return exprNodeType::rightUnary;
    }

    exprNode* rightUnaryOpNode::clone() const {
      return new rightUnaryOpNode(token,
                                  (const unaryOperator_t&) op,
                                  *value);
    }

    bool rightUnaryOpNode::canEvaluate() const {
      return value->canEvaluate();
    }

    primitive rightUnaryOpNode::evaluate() const {
      primitive pValue = value->evaluate();
      return ((unaryOperator_t&) op)(pValue);
    }

    exprNode* rightUnaryOpNode::startNode() {
      return value->startNode();
    }

    void rightUnaryOpNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
    }

    variable_t* rightUnaryOpNode::getVariable() {
      return value->getVariable();
    }

    void rightUnaryOpNode::print(printer &pout) const {
      pout << *value << op;
    }

    void rightUnaryOpNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << op;
      std::cerr << "] (rightUnary)\n";
      value->childDebugPrint(prefix);
    }

    binaryOpNode::binaryOpNode(token_t *token_,
                               const binaryOperator_t &op_,
                               const exprNode &leftValue_,
                               const exprNode &rightValue_) :
      exprOpNode(token_, op_),
      leftValue(leftValue_.clone()),
      rightValue(rightValue_.clone()) {}

    binaryOpNode::binaryOpNode(const binaryOpNode &node) :
      exprOpNode(node.token, node.op),
      leftValue(node.leftValue->clone()),
      rightValue(node.rightValue->clone()) {}

    binaryOpNode::~binaryOpNode() {
      delete leftValue;
      delete rightValue;
    }

    udim_t binaryOpNode::type() const {
      return exprNodeType::binary;
    }

    exprNode* binaryOpNode::clone() const {
      return new binaryOpNode(token,
                              (const binaryOperator_t&) op,
                              *leftValue,
                              *rightValue);
    }

    bool binaryOpNode::canEvaluate() const {
      if (op.opType & (operatorType::scope     |
                       operatorType::dot       |
                       operatorType::dotStar   |
                       operatorType::arrow     |
                       operatorType::arrowStar)) {
        return false;
      }
      return (leftValue->canEvaluate() &&
              rightValue->canEvaluate());
    }

    primitive binaryOpNode::evaluate() const {
      primitive pLeft  = leftValue->evaluate();
      primitive pRight = rightValue->evaluate();
      return ((binaryOperator_t&) op)(pLeft, pRight);
    }

    exprNode* binaryOpNode::startNode() {
      return leftValue->startNode();
    }

    exprNode* binaryOpNode::endNode() {
      return rightValue->endNode();
    }

    void binaryOpNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&leftValue);
      children.push_back(&rightValue);
    }

    variable_t* binaryOpNode::getVariable() {
      return leftValue->getVariable();
    }

    void binaryOpNode::print(printer &pout) const {
      pout << *leftValue
           << ' ' << op
           << ' ' << *rightValue;
    }

    void binaryOpNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << op;
      std::cerr << "] (binary)\n";
      leftValue->childDebugPrint(prefix);
      rightValue->childDebugPrint(prefix);
    }

    ternaryOpNode::ternaryOpNode(const exprNode &checkValue_,
                                 const exprNode &trueValue_,
                                 const exprNode &falseValue_) :
      exprOpNode(checkValue_.token, op::ternary),
      checkValue(checkValue_.clone()),
      trueValue(trueValue_.clone()),
      falseValue(falseValue_.clone()) {}

    ternaryOpNode::ternaryOpNode(const ternaryOpNode &node) :
      exprOpNode(node.token, op::ternary),
      checkValue(node.checkValue->clone()),
      trueValue(node.trueValue->clone()),
      falseValue(node.falseValue->clone()) {}

    ternaryOpNode::~ternaryOpNode() {
      delete checkValue;
      delete trueValue;
      delete falseValue;
    }

    udim_t ternaryOpNode::type() const {
      return exprNodeType::ternary;
    }

    opType_t ternaryOpNode::opType() const {
      return operatorType::ternary;
    }

    exprNode* ternaryOpNode::clone() const {
      return new ternaryOpNode(*checkValue,
                               *trueValue,
                               *falseValue);
    }

    bool ternaryOpNode::canEvaluate() const {
      return (checkValue->canEvaluate() &&
              trueValue->canEvaluate()  &&
              falseValue->canEvaluate());
    }

    primitive ternaryOpNode::evaluate() const {
      if ((bool) checkValue->evaluate()) {
        return trueValue->evaluate();
      }
      return falseValue->evaluate();
    }

    exprNode* ternaryOpNode::startNode() {
      return checkValue->startNode();
    }

    exprNode* ternaryOpNode::endNode() {
      return falseValue->endNode();
    }

    void ternaryOpNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&checkValue);
      children.push_back(&trueValue);
      children.push_back(&falseValue);
    }

    void ternaryOpNode::print(printer &pout) const {
      pout << *checkValue
           << " ? " << *trueValue
           << " : " << *falseValue;
    }

    void ternaryOpNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[?:] (ternary)\n";
      checkValue->childDebugPrint(prefix);
      trueValue->childDebugPrint(prefix);
      falseValue->childDebugPrint(prefix);
    }
    //==================================

    //---[ Pseudo Operators ]-----------
    subscriptNode::subscriptNode(token_t *token_,
                                 const exprNode &value_,
                                 const exprNode &index_) :
      exprNode(token_),
      value(value_.clone()),
      index(index_.clone()) {}

    subscriptNode::subscriptNode(const subscriptNode &node) :
      exprNode(node.token),
      value(node.value->clone()),
      index(node.index->clone()) {}

    subscriptNode::~subscriptNode() {
      delete value;
      delete index;
    }

    udim_t subscriptNode::type() const {
      return exprNodeType::subscript;
    }

    exprNode* subscriptNode::clone() const {
      return new subscriptNode(token, *value, *index);
    }

    exprNode* subscriptNode::startNode() {
      return value->startNode();
    }

    exprNode* subscriptNode::endNode() {
      return index->endNode();
    }

    void subscriptNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
      children.push_back(&index);
    }

    variable_t* subscriptNode::getVariable() {
      return value->getVariable();
    }

    void subscriptNode::print(printer &pout) const {
      pout << *value
           << '[' << *index << ']';
    }

    void subscriptNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << *index;
      std::cerr << "] (subscript)\n";
      value->childDebugPrint(prefix);
    }

    callNode::callNode(token_t *token_,
                       const exprNode &value_,
                       const exprNodeVector &args_) :
      exprNode(token_),
      value(value_.clone()) {
      cloneExprNodeVector(args, args_);
    }

    callNode::callNode(const callNode &node) :
      exprNode(node.token),
      value(node.value->clone()) {
      cloneExprNodeVector(args, node.args);
    }

    callNode::~callNode() {
      delete value;
      freeExprNodeVector(args);
    }

    udim_t callNode::type() const {
      return exprNodeType::call;
    }

    exprNode* callNode::clone() const {
      return new callNode(token, *value, args);
    }

    exprNode* callNode::startNode() {
      return value->startNode();
    }

    exprNode* callNode::endNode() {
      const int argCount = (int) args.size();
      if (!argCount) {
        return value->endNode();
      }
      return args[argCount - 1]->endNode();
    }

    void callNode::setChildren(exprNodeRefVector &children) {
      const int argCount = (int) args.size();
      children.reserve(1 + argCount);

      children.push_back(&value);
      for (int i = 0; i < argCount; ++i) {
        children.push_back(&(args[i]));
      }
    }

    void callNode::print(printer &pout) const {
      pout << *value
           << '(';
      const int argCount = (int) args.size();
      for (int i = 0; i < argCount; ++i) {
        if (i) {
          pout << ", ";
        }
        pout << *(args[i]);
      }
      pout << ')';
    }

    void callNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << *value;
      std::cerr << "] (call)\n";
      for (int i = 0; i < ((int) args.size()); ++i) {
        args[i]->childDebugPrint(prefix);
      }
    }

    newNode::newNode(token_t *token_,
                     const vartype_t &valueType_,
                     const exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()),
      size(noExprNode.clone()) {}

    newNode::newNode(token_t *token_,
                     const vartype_t &valueType_,
                     const exprNode &value_,
                     const exprNode &size_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()),
      size(size_.clone()) {}

    newNode::newNode(const newNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value->clone()),
      size(node.size->clone()) {}

    newNode::~newNode() {
      delete value;
      delete size;
    }

    udim_t newNode::type() const {
      return exprNodeType::new_;
    }

    exprNode* newNode::clone() const {
      return new newNode(token, valueType, *value, *size);
    }

    exprNode* newNode::endNode() {
      return (size ? size : value)->endNode();
    }

    void newNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
      children.push_back(&size);
    }

    exprNode* newNode::wrapInParentheses() {
      return new parenthesesNode(token, *this);
    }

    void newNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "new " << valueType << *value;
      if (size->type() != exprNodeType::empty) {
        pout << '[' << *size << ']';
      }
    }

    void newNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << valueType;
      std::cerr << "] (new)\n";
      value->childDebugPrint(prefix);
      size->childDebugPrint(prefix);
    }

    deleteNode::deleteNode(token_t *token_,
                           const exprNode &value_,
                           const bool isArray_) :
      exprNode(token_),
      value(value_.clone()),
      isArray(isArray_) {}

    deleteNode::deleteNode(const deleteNode &node) :
      exprNode(node.token),
      value(node.value->clone()),
      isArray(node.isArray) {}

    deleteNode::~deleteNode() {
      delete value;
    }

    udim_t deleteNode::type() const {
      return exprNodeType::delete_;
    }

    exprNode* deleteNode::clone() const {
      return new deleteNode(token, *value, isArray);
    }

    exprNode* deleteNode::endNode() {
      return value->endNode();
    }

    void deleteNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
    }

    exprNode* deleteNode::wrapInParentheses() {
      return new parenthesesNode(token, *this);
    }

    void deleteNode::print(printer &pout) const {
      pout << "delete ";
      if (isArray) {
        pout << "[] ";
      }
      pout << *value;
    }

    void deleteNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << '\n'
                << prefix << "|---[";
      pout << *value;
      std::cerr << "] (delete";
      if (isArray) {
        std::cerr << " []";
      }
      std::cerr << ")\n";
    }

    throwNode::throwNode(token_t *token_,
                         const exprNode &value_) :
      exprNode(token_),
      value(value_.clone()) {}

    throwNode::throwNode(const throwNode &node) :
      exprNode(node.token),
      value(node.value->clone()) {}

    throwNode::~throwNode() {
      delete value;
    }

    udim_t throwNode::type() const {
      return exprNodeType::throw_;
    }

    exprNode* throwNode::clone() const {
      return new throwNode(token, *value);
    }

    exprNode* throwNode::endNode() {
      return value->endNode();
    }

    void throwNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
    }

    exprNode* throwNode::wrapInParentheses() {
      return new parenthesesNode(token, *this);
    }

    void throwNode::print(printer &pout) const {
      pout << "throw";
      if (value->type() != exprNodeType::empty) {
        pout << ' ' << *value;
      }
    }

    void throwNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|\n"
                << prefix << "|---[";
      pout << *value;
      std::cerr << "] (throw)\n";
    }
    //==================================

    //---[ Builtins ]-------------------
    sizeofNode::sizeofNode(token_t *token_,
                           const exprNode &value_) :
      exprNode(token_),
      value(value_.clone()) {}

    sizeofNode::sizeofNode(const sizeofNode &node) :
      exprNode(node.token),
      value(node.value->clone()) {}

    sizeofNode::~sizeofNode() {
      delete value;
    }

    udim_t sizeofNode::type() const {
      return exprNodeType::sizeof_;
    }

    exprNode* sizeofNode::endNode() {
      return value->endNode();
    }

    exprNode* sizeofNode::clone() const {
      return new sizeofNode(token, *value);
    }

    bool sizeofNode::canEvaluate() const {
      return value->canEvaluate();
    }

    primitive sizeofNode::evaluate() const {
      return value->evaluate().sizeof_();
    }

    void sizeofNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
    }

    void sizeofNode::print(printer &pout) const {
      pout << "sizeof(" << *value << ')';
    }

    void sizeofNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << '\n'
                << prefix << "|---[";
      pout << *value;
      std::cerr << "] (sizeof)\n";
    }

    funcCastNode::funcCastNode(token_t *token_,
                               const vartype_t &valueType_,
                               const exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    funcCastNode::funcCastNode(const funcCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value->clone()) {}

    funcCastNode::~funcCastNode() {
      delete value;
    }

    udim_t funcCastNode::type() const {
      return exprNodeType::funcCast;
    }

    exprNode* funcCastNode::startNode() {
      return value->startNode();
    }

    exprNode* funcCastNode::endNode() {
      return value->endNode();
    }

    exprNode* funcCastNode::clone() const {
      return new funcCastNode(token, valueType, *value);
    }

    void funcCastNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
    }

    void funcCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << valueType << '(' << *value << ')';
    }

    void funcCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << valueType;
      std::cerr << "] (funcCast)\n";
      value->childDebugPrint(prefix);
    }

    parenCastNode::parenCastNode(token_t *token_,
                                 const vartype_t &valueType_,
                                 const exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    parenCastNode::parenCastNode(const parenCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value->clone()) {}

    parenCastNode::~parenCastNode() {
      delete value;
    }

    udim_t parenCastNode::type() const {
      return exprNodeType::parenCast;
    }

    exprNode* parenCastNode::startNode() {
      return value->startNode();
    }

    exprNode* parenCastNode::endNode() {
      return value->endNode();
    }

    exprNode* parenCastNode::clone() const {
      return new parenCastNode(token, valueType, *value);
    }

    void parenCastNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
    }

    exprNode* parenCastNode::wrapInParentheses() {
      return new parenthesesNode(token, *this);
    }

    void parenCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << '(' << valueType << ") " << *value;
    }

    void parenCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << valueType;
      std::cerr << "] (parenCast)\n";
      value->childDebugPrint(prefix);
    }

    constCastNode::constCastNode(token_t *token_,
                                 const vartype_t &valueType_,
                                 const exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    constCastNode::constCastNode(const constCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value->clone()) {}

    constCastNode::~constCastNode() {
      delete value;
    }

    udim_t constCastNode::type() const {
      return exprNodeType::constCast;
    }

    exprNode* constCastNode::endNode() {
      return value->endNode();
    }

    exprNode* constCastNode::clone() const {
      return new constCastNode(token, valueType, *value);
    }

    void constCastNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
    }

    void constCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "const_cast<" << valueType << ">("
           << *value << ')';
    }

    void constCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << valueType;
      std::cerr << "] (constCast)\n";
      value->childDebugPrint(prefix);
    }

    staticCastNode::staticCastNode(token_t *token_,
                                   const vartype_t &valueType_,
                                   const exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    staticCastNode::staticCastNode(const staticCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value->clone()) {}

    staticCastNode::~staticCastNode() {
      delete value;
    }

    udim_t staticCastNode::type() const {
      return exprNodeType::staticCast;
    }

    exprNode* staticCastNode::endNode() {
      return value->endNode();
    }

    exprNode* staticCastNode::clone() const {
      return new staticCastNode(token, valueType, *value);
    }

    void staticCastNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
    }

    void staticCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "static_cast<" << valueType << ">("
           << *value << ')';
    }

    void staticCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << valueType;
      std::cerr << "] (staticCast)\n";
      value->childDebugPrint(prefix);
    }

    reinterpretCastNode::reinterpretCastNode(token_t *token_,
                                             const vartype_t &valueType_,
                                             const exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    reinterpretCastNode::reinterpretCastNode(const reinterpretCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value->clone()) {}

    reinterpretCastNode::~reinterpretCastNode() {
      delete value;
    }

    udim_t reinterpretCastNode::type() const {
      return exprNodeType::reinterpretCast;
    }

    exprNode* reinterpretCastNode::endNode() {
      return value->endNode();
    }

    exprNode* reinterpretCastNode::clone() const {
      return new reinterpretCastNode(token, valueType, *value);
    }

    void reinterpretCastNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
    }

    void reinterpretCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "reinterpret_cast<" << valueType << ">("
           << *value << ')';
    }

    void reinterpretCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << valueType;
      std::cerr << "] (reinterpretCast)\n";
      value->childDebugPrint(prefix);
    }

    dynamicCastNode::dynamicCastNode(token_t *token_,
                                     const vartype_t &valueType_,
                                     const exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    dynamicCastNode::dynamicCastNode(const dynamicCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value->clone()) {}

    dynamicCastNode::~dynamicCastNode() {
      delete value;
    }

    udim_t dynamicCastNode::type() const {
      return exprNodeType::dynamicCast;
    }

    exprNode* dynamicCastNode::endNode() {
      return value->endNode();
    }

    exprNode* dynamicCastNode::clone() const {
      return new dynamicCastNode(token, valueType, *value);
    }

    void dynamicCastNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
    }

    void dynamicCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "dynamic_cast<" << valueType << ">("
           << *value << ')';
    }

    void dynamicCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << valueType;
      std::cerr << "] (dynamicCast)\n";
      value->childDebugPrint(prefix);
    }
    //==================================

    //---[ Misc ]-----------------------
    pairNode::pairNode(operatorToken &opToken,
                       const exprNode &value_) :
      exprNode(&opToken),
      op(*(opToken.op)),
      value(value_.clone()) {}

    pairNode::pairNode(const pairNode &node) :
      exprNode(node.token),
      op(node.op),
      value(node.value->clone()) {}

    pairNode::~pairNode() {
      delete value;
    }

    udim_t pairNode::type() const {
      return exprNodeType::pair;
    }

    opType_t pairNode::opType() const {
      return op.opType;
    }

    exprNode* pairNode::startNode() {
      return value->startNode();
    }

    exprNode* pairNode::endNode() {
      return value->endNode();
    }

    exprNode* pairNode::clone() const {
      return new pairNode(token->to<operatorToken>(),
                          *value);
    }

    bool pairNode::canEvaluate() const {
      token->printError("[Waldo] (pairNode) Unsure how you got here...");
      return false;
    }

    primitive pairNode::evaluate() const {
      token->printError("[Waldo] (pairNode) Unsure how you got here...");
      return primitive();
    }

    void pairNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
    }

    void pairNode::print(printer &pout) const {
      token->printError("[Waldo] (pairNode) Unsure how you got here...");
    }

    void pairNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      pout << op;
      std::cerr << "] (pairNode)\n";
      value->childDebugPrint(prefix);
    }

    parenthesesNode::parenthesesNode(token_t *token_,
                                     const exprNode &value_) :
      exprNode(token_),
      value(value_.clone()) {}

    parenthesesNode::parenthesesNode(const parenthesesNode &node) :
      exprNode(node.token),
      value(node.value->clone()) {}

    parenthesesNode::~parenthesesNode() {
      delete value;
    }

    udim_t parenthesesNode::type() const {
      return exprNodeType::parentheses;
    }

    exprNode* parenthesesNode::startNode() {
      return value->startNode();
    }

    exprNode* parenthesesNode::endNode() {
      return value->endNode();
    }

    exprNode* parenthesesNode::clone() const {
      return new parenthesesNode(token, *value);
    }

    bool parenthesesNode::canEvaluate() const {
      return value->canEvaluate();
    }

    primitive parenthesesNode::evaluate() const {
      return value->evaluate();
    }

    void parenthesesNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
    }

    variable_t* parenthesesNode::getVariable() {
      return value->getVariable();
    }

    void parenthesesNode::print(printer &pout) const {
      pout << '(' << *value << ')';
    }

    void parenthesesNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[()] (parentheses)\n";
      value->childDebugPrint(prefix);
    }

    tupleNode::tupleNode(token_t *token_,
                         const exprNodeVector &args_) :
      exprNode(token_) {
      cloneExprNodeVector(args, args_);
    }

    tupleNode::tupleNode(const tupleNode &node) :
      exprNode(node.token) {
      cloneExprNodeVector(args, node.args);
    }

    tupleNode::~tupleNode() {
      freeExprNodeVector(args);
    }

    udim_t tupleNode::type() const {
      return exprNodeType::tuple;
    }

    exprNode* tupleNode::startNode() {
      const int argCount = (int) args.size();
      return (argCount ? args[0]->startNode() : this);
    }

    exprNode* tupleNode::endNode() {
      const int argCount = (int) args.size();
      return (argCount ? args[argCount - 1]->endNode() : this);
    }

    exprNode* tupleNode::clone() const {
      return new tupleNode(token, args);
    }

    void tupleNode::setChildren(exprNodeRefVector &children) {
      const int argCount = (int) args.size();
      if (!argCount) {
        return;
      }

      children.reserve(argCount);
      for (int i = 0; i < argCount; ++i) {
        children.push_back(&(args[i]));
      }
    }

    void tupleNode::print(printer &pout) const {
      pout << '{';
      const int argCount = (int) args.size();
      for (int i = 0; i < argCount; ++i) {
        if (i) {
          pout << ", ";
        }
        pout << *(args[i]);
      }
      pout << '}';
    }

    void tupleNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---(tuple)\n";
      for (int i = 0; i < ((int) args.size()); ++i) {
        args[i]->childDebugPrint(prefix);
      }
    }
    //==================================

    //---[ Extensions ]-----------------
    cudaCallNode::cudaCallNode(token_t *token_,
                               const exprNode &value_,
                               const exprNode &blocks_,
                               const exprNode &threads_) :
      exprNode(token_),
      value(value_.clone()),
      blocks(blocks_.clone()),
      threads(threads_.clone()) {}

    cudaCallNode::cudaCallNode(const cudaCallNode &node) :
      exprNode(node.token),
      value(node.value->clone()),
      blocks(node.blocks->clone()),
      threads(node.threads->clone()) {}

    cudaCallNode::~cudaCallNode() {
      delete value;
      delete blocks;
      delete threads;
    }

    udim_t cudaCallNode::type() const {
      return exprNodeType::cudaCall;
    }

    exprNode* cudaCallNode::startNode() {
      return value->startNode();
    }

    exprNode* cudaCallNode::endNode() {
      return threads->endNode();
    }

    exprNode* cudaCallNode::clone() const {
      return new cudaCallNode(token,
                              *value,
                              *blocks,
                              *threads);
    }

    void cudaCallNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
      children.push_back(&blocks);
      children.push_back(&threads);
    }

    void cudaCallNode::print(printer &pout) const {
      pout << *value
           << "<<<"
           << *blocks << ", " << *threads
           << ">>>";
    }

    void cudaCallNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[<<<...>>>";
      std::cerr << "] (cudaCall)\n";
      value->childDebugPrint(prefix);
      blocks->childDebugPrint(prefix);
      threads->childDebugPrint(prefix);
    }
    //==================================
  }
}
