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
#include "expression.hpp"
#include "occa/tools/string.hpp"

namespace occa {
  namespace lang {
    namespace exprNodeType {
      const int empty           = (1 << 0);
      const int primitive       = (1 << 1);
      const int char_           = (1 << 2);
      const int string          = (1 << 3);
      const int identifier      = (1 << 4);
      const int variable        = (1 << 5);

      const int value           = (primitive |
                                   variable);

      const int leftUnary       = (1 << 6);
      const int rightUnary      = (1 << 7);
      const int binary          = (1 << 8);
      const int ternary         = (1 << 9);
      const int op              = (leftUnary  |
                                   rightUnary |
                                   binary     |
                                   ternary);

      const int pair            = (1 << 10);
      const int subscript       = (1 << 11);
      const int call            = (1 << 12);
      const int new_            = (1 << 13);
      const int delete_         = (1 << 14);
      const int throw_          = (1 << 15);
      const int sizeof_         = (1 << 16);
      const int funcCast        = (1 << 17);
      const int parenCast       = (1 << 18);
      const int constCast       = (1 << 19);
      const int staticCast      = (1 << 20);
      const int reinterpretCast = (1 << 21);
      const int dynamicCast     = (1 << 22);
      const int parentheses     = (1 << 23);
      const int tuple           = (1 << 24);
      const int cudaCall        = (1 << 25);
    }

    exprNode::exprNode(token_t *token_) :
      token(token_
            ? token_->clone()
            : NULL) {}

    exprNode::~exprNode() {
      delete token;
    }

    bool exprNode::canEvaluate() const {
      return false;
    }

    primitive exprNode::evaluate() const {
      return primitive();
    }

    std::string exprNode::toString() const {
      std::stringstream ss;
      printer pout(ss);
      print(pout);
      return ss.str();
    }

    void exprNode::debugPrint() const {
      debugPrint("");
      std::cerr << '\n';
    }

    void exprNode::childDebugPrint(const std::string &prefix) const {
      debugPrint(prefix + "|   ");
    }

    void cloneExprNodeVector(exprNodeVector &dest,
                             const exprNodeVector &src) {
      const int nodes = (int) src.size();
      dest.clear();
      dest.reserve(nodes);
      for (int i = 0; i < nodes; ++i) {
        dest.push_back(&(src[i]->clone()));
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

    int emptyNode::type() const {
      return exprNodeType::empty;
    }

    exprNode& emptyNode::clone() const {
      return *(new emptyNode());
    }

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

    int primitiveNode::type() const {
      return exprNodeType::primitive;
    }

    exprNode& primitiveNode::clone() const {
      return *(new primitiveNode(token, value));
    }

    bool primitiveNode::canEvaluate() const {
      return true;
    }

    primitive primitiveNode::evaluate() const {
      return value;
    }

    void primitiveNode::print(printer &pout) const {
      pout << (std::string) value;
    }

    void primitiveNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      print(pout);
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

    int charNode::type() const {
      return exprNodeType::char_;
    }

    exprNode& charNode::clone() const {
      return *(new charNode(token, value));
    }

    void charNode::print(printer &pout) const {
      pout << "'" << escape(value, '\'') << '"';
    }

    void charNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << '\n'
                << prefix << "|---[";
      print(pout);
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

    int stringNode::type() const {
      return exprNodeType::string;
    }

    exprNode& stringNode::clone() const {
      return *(new stringNode(token, value));
    }

    void stringNode::print(printer &pout) const {
      pout << "\"" << escape(value, '"') << "\"";
    }

    void stringNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      print(pout);
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

    int identifierNode::type() const {
      return exprNodeType::identifier;
    }

    exprNode& identifierNode::clone() const {
      return *(new identifierNode(token, value));
    }

    void identifierNode::print(printer &pout) const {
      pout << value;
    }

    void identifierNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << '\n'
                << prefix << "|---[";
      print(pout);
      std::cerr << "] (identifier)\n";
    }
    //  |===============================

    //  |---[ Variable ]----------------
    variableNode::variableNode(token_t *token_,
                               variable &value_) :
      exprNode(token_),
      value(value_) {}

    variableNode::variableNode(const variableNode &node) :
      exprNode(node.token),
      value(node.value) {}

    variableNode::~variableNode() {}

    int variableNode::type() const {
      return exprNodeType::variable;
    }

    exprNode& variableNode::clone() const {
      return *(new variableNode(token, value));
    }

    void variableNode::print(printer &pout) const {
      value.print(pout);
    }

    void variableNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      print(pout);
      std::cerr << "] (variable)\n";
    }
    //  |===============================
    //==================================

    //---[ Operators ]------------------
    leftUnaryOpNode::leftUnaryOpNode(token_t *token_,
                                     const unaryOperator_t &op_,
                                     exprNode &value_) :
      exprNode(token_),
      op(op_),
      value(value_.clone()) {}

    leftUnaryOpNode::leftUnaryOpNode(const leftUnaryOpNode &node) :
      exprNode(node.token),
      op(node.op),
      value(node.value.clone()) {}

    leftUnaryOpNode::~leftUnaryOpNode() {
      delete &value;
    }

    int leftUnaryOpNode::type() const {
      return exprNodeType::leftUnary;
    }

    opType_t leftUnaryOpNode::optype() const {
      return op.opType;
    }

    exprNode& leftUnaryOpNode::clone() const {
      return *(new leftUnaryOpNode(token, op, value));
    }

    bool leftUnaryOpNode::canEvaluate() const {
      if (op.opType & (operatorType::dereference |
                       operatorType::address)) {
        return false;
      }
      return value.canEvaluate();
    }

    primitive leftUnaryOpNode::evaluate() const {
      primitive pValue = value.evaluate();
      return op(pValue);
    }

    void leftUnaryOpNode::print(printer &pout) const {
      op.print(pout);
      value.print(pout);
    }

    void leftUnaryOpNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      op.print(pout);
      std::cerr << "] (leftUnary)\n";
      value.childDebugPrint(prefix);
    }

    rightUnaryOpNode::rightUnaryOpNode(token_t *token_,
                                       const unaryOperator_t &op_,
                                       exprNode &value_) :
      exprNode(token_),
      op(op_),
      value(value_.clone()) {}

    rightUnaryOpNode::rightUnaryOpNode(const rightUnaryOpNode &node) :
      exprNode(node.token),
      op(node.op),
      value(node.value.clone()) {}

    rightUnaryOpNode::~rightUnaryOpNode() {
      delete &value;
    }

    int rightUnaryOpNode::type() const {
      return exprNodeType::rightUnary;
    }

    opType_t rightUnaryOpNode::optype() const {
      return op.opType;
    }

    exprNode& rightUnaryOpNode::clone() const {
      return *(new rightUnaryOpNode(token, op, value));
    }

    bool rightUnaryOpNode::canEvaluate() const {
      return value.canEvaluate();
    }

    primitive rightUnaryOpNode::evaluate() const {
      primitive pValue = value.evaluate();
      return op(pValue);
    }

    void rightUnaryOpNode::print(printer &pout) const {
      value.print(pout);
      op.print(pout);
    }

    void rightUnaryOpNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      op.print(pout);
      std::cerr << "] (rightUnary)\n";
      value.childDebugPrint(prefix);
    }

    binaryOpNode::binaryOpNode(token_t *token_,
                               const binaryOperator_t &op_,
                               exprNode &leftValue_,
                               exprNode &rightValue_) :
      exprNode(token_),
      op(op_),
      leftValue(leftValue_.clone()),
      rightValue(rightValue_.clone()) {}

    binaryOpNode::binaryOpNode(const binaryOpNode &node) :
      exprNode(node.token),
      op(node.op),
      leftValue(node.leftValue.clone()),
      rightValue(node.rightValue.clone()) {}

    binaryOpNode::~binaryOpNode() {
      delete &leftValue;
      delete &rightValue;
    }

    int binaryOpNode::type() const {
      return exprNodeType::binary;
    }

    opType_t binaryOpNode::optype() const {
      return op.opType;
    }

    exprNode& binaryOpNode::clone() const {
      return *(new binaryOpNode(token, op, leftValue, rightValue));
    }

    bool binaryOpNode::canEvaluate() const {
      if (op.opType & (operatorType::scope     |
                       operatorType::dot       |
                       operatorType::dotStar   |
                       operatorType::arrow     |
                       operatorType::arrowStar)) {
        return false;
      }
      return (leftValue.canEvaluate() &&
              rightValue.canEvaluate());
    }

    primitive binaryOpNode::evaluate() const {
      primitive pLeft  = leftValue.evaluate();
      primitive pRight = rightValue.evaluate();
      return op(pLeft, pRight);
    }

    void binaryOpNode::print(printer &pout) const {
      leftValue.print(pout);
      pout << ' ';
      op.print(pout);
      pout << ' ';
      rightValue.print(pout);
    }

    void binaryOpNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      op.print(pout);
      std::cerr << "] (binary)\n";
      leftValue.childDebugPrint(prefix);
      rightValue.childDebugPrint(prefix);
    }

    ternaryOpNode::ternaryOpNode(token_t *token_,
                                 exprNode &checkValue_,
                                 exprNode &trueValue_,
                                 exprNode &falseValue_) :
      exprNode(token_),
      checkValue(checkValue_.clone()),
      trueValue(trueValue_.clone()),
      falseValue(falseValue_.clone()) {}

    ternaryOpNode::ternaryOpNode(const ternaryOpNode &node) :
      exprNode(node.token),
      checkValue(node.checkValue.clone()),
      trueValue(node.trueValue.clone()),
      falseValue(node.falseValue.clone()) {}

    ternaryOpNode::~ternaryOpNode() {
      delete &checkValue;
      delete &trueValue;
      delete &falseValue;
    }

    int ternaryOpNode::type() const {
      return exprNodeType::ternary;
    }

    opType_t ternaryOpNode::optype() const {
      return operatorType::ternary;
    }

    exprNode& ternaryOpNode::clone() const {
      return *(new ternaryOpNode(token,
                                 checkValue,
                                 trueValue,
                                 falseValue));
    }

    bool ternaryOpNode::canEvaluate() const {
      return (checkValue.canEvaluate() &&
              trueValue.canEvaluate()  &&
              falseValue.canEvaluate());
    }

    primitive ternaryOpNode::evaluate() const {
      if ((bool) checkValue.evaluate()) {
        return trueValue.evaluate();
      }
      return falseValue.evaluate();
    }

    void ternaryOpNode::print(printer &pout) const {
      checkValue.print(pout);
      pout << " ? ";
      trueValue.print(pout);
      pout << " : ";
      falseValue.print(pout);
    }

    void ternaryOpNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[?:] (ternary)\n";
      checkValue.childDebugPrint(prefix);
      trueValue.childDebugPrint(prefix);
      falseValue.childDebugPrint(prefix);
    }
    //==================================

    //---[ Pseudo Operators ]-----------
    subscriptNode::subscriptNode(token_t *token_,
                                 exprNode &value_,
                                 exprNode &index_) :
      exprNode(token_),
      value(value_.clone()),
      index(index_.clone()) {}

    subscriptNode::subscriptNode(const subscriptNode &node) :
      exprNode(node.token),
      value(node.value.clone()),
      index(node.index.clone()) {}

    subscriptNode::~subscriptNode() {
      delete &value;
      delete &index;
    }

    int subscriptNode::type() const {
      return exprNodeType::subscript;
    }

    exprNode& subscriptNode::clone() const {
      return *(new subscriptNode(token, value, index));
    }

    void subscriptNode::print(printer &pout) const {
      value.print(pout);
      pout << '[';
      index.print(pout);
      pout << ']';
    }

    void subscriptNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      index.print(pout);
      std::cerr << "] (subscript)\n";
      value.childDebugPrint(prefix);
    }

    callNode::callNode(token_t *token_,
                       exprNode &value_,
                       const exprNodeVector &args_) :
      exprNode(token_),
      value(value_.clone()) {
      cloneExprNodeVector(args, args_);
    }

    callNode::callNode(const callNode &node) :
      exprNode(node.token),
      value(node.value.clone()) {
      cloneExprNodeVector(args, node.args);
    }

    callNode::~callNode() {
      delete &value;
      freeExprNodeVector(args);
    }

    int callNode::type() const {
      return exprNodeType::call;
    }

    exprNode& callNode::clone() const {
      return *(new callNode(token, value, args));
    }

    void callNode::print(printer &pout) const {
      value.print(pout);
      pout << '(';
      const int argCount = (int) args.size();
      for (int i = 0; i < argCount; ++i) {
        if (i) {
          pout << ", ";
        }
        args[i]->print(pout);
      }
      pout << ')';
    }

    void callNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      value.print(pout);
      std::cerr << "] (call)\n";
      for (int i = 0; i < ((int) args.size()); ++i) {
        args[i]->childDebugPrint(prefix);
      }
    }

    newNode::newNode(token_t *token_,
                     type_t &valueType_,
                     exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()),
      size(noExprNode.clone()) {}


    newNode::newNode(token_t *token_,
                     type_t &valueType_,
                     exprNode &value_,
                     exprNode &size_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()),
      size(size_.clone()) {}

    newNode::newNode(const newNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value.clone()),
      size(node.size.clone()) {}

    newNode::~newNode() {
      delete &value;
      delete &size;
    }

    int newNode::type() const {
      return exprNodeType::new_;
    }

    exprNode& newNode::clone() const {
      return *(new newNode(token, valueType, value, size));
    }

    void newNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "new ";
      valueType.print(pout);
      value.print(pout);
      if (size.type() != exprNodeType::empty) {
        pout << '[';
        size.print(pout);
        pout << ']';
      }
    }

    void newNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      valueType.print(pout);
      std::cerr << "] (new)\n";
      value.childDebugPrint(prefix);
      size.childDebugPrint(prefix);
    }

    deleteNode::deleteNode(token_t *token_,
                           exprNode &value_,
                           const bool isArray_) :
      exprNode(token_),
      value(value_.clone()),
      isArray(isArray_) {}

    deleteNode::deleteNode(const deleteNode &node) :
      exprNode(node.token),
      value(node.value.clone()),
      isArray(node.isArray) {}

    deleteNode::~deleteNode() {
      delete &value;
    }

    int deleteNode::type() const {
      return exprNodeType::delete_;
    }

    exprNode& deleteNode::clone() const {
      return *(new deleteNode(token, value, isArray));
    }

    void deleteNode::print(printer &pout) const {
      pout << "delete ";
      if (isArray) {
        pout << "[] ";
      }
      value.print(pout);
    }

    void deleteNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << '\n'
                << prefix << "|---[";
      value.print(pout);
      std::cerr << "] (delete";
      if (isArray) {
        std::cerr << " []";
      }
      std::cerr << ")\n";
    }

    throwNode::throwNode(token_t *token_,
                         exprNode &value_) :
      exprNode(token_),
      value(value_.clone()) {}

    throwNode::throwNode(const throwNode &node) :
      exprNode(node.token),
      value(node.value.clone()) {}

    throwNode::~throwNode() {
      delete &value;
    }

    int throwNode::type() const {
      return exprNodeType::throw_;
    }

    exprNode& throwNode::clone() const {
      return *(new throwNode(token, value));
    }

    void throwNode::print(printer &pout) const {
      pout << "throw";
      if (value.type() != exprNodeType::empty) {
        pout << ' ';
        value.print(pout);
      }
    }

    void throwNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|\n"
                << prefix << "|---[";
      value.print(pout);
      std::cerr << "] (throw)\n";
    }
    //==================================

    //---[ Builtins ]-------------------
    sizeofNode::sizeofNode(token_t *token_,
                           exprNode &value_) :
      exprNode(token_),
      value(value_.clone()) {}

    sizeofNode::sizeofNode(const sizeofNode &node) :
      exprNode(node.token),
      value(node.value.clone()) {}

    sizeofNode::~sizeofNode() {
      delete &value;
    }

    int sizeofNode::type() const {
      return exprNodeType::sizeof_;
    }

    exprNode& sizeofNode::clone() const {
      return *(new sizeofNode(token, value));
    }

    bool sizeofNode::canEvaluate() const {
      return value.canEvaluate();
    }

    primitive sizeofNode::evaluate() const {
      return value.evaluate().sizeof_();
    }

    void sizeofNode::print(printer &pout) const {
      pout << "sizeof(";
      value.print(pout);
      pout << ')';
    }

    void sizeofNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << '\n'
                << prefix << "|---[";
      value.print(pout);
      std::cerr << "] (sizeof)\n";
    }

    funcCastNode::funcCastNode(token_t *token_,
                               type_t &valueType_,
                               exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    funcCastNode::funcCastNode(const funcCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value.clone()) {}

    funcCastNode::~funcCastNode() {
      delete &value;
    }

    int funcCastNode::type() const {
      return exprNodeType::funcCast;
    }

    exprNode& funcCastNode::clone() const {
      return *(new funcCastNode(token, valueType, value));
    }

    void funcCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      valueType.print(pout);
      pout << '(';
      value.print(pout);
      pout << ')';
    }

    void funcCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      valueType.print(pout);
      std::cerr << "] (funcCast)\n";
      value.childDebugPrint(prefix);
    }

    parenCastNode::parenCastNode(token_t *token_,
                                 type_t &valueType_,
                                 exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    parenCastNode::parenCastNode(const parenCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value.clone()) {}

    parenCastNode::~parenCastNode() {
      delete &value;
    }

    int parenCastNode::type() const {
      return exprNodeType::parenCast;
    }

    exprNode& parenCastNode::clone() const {
      return *(new parenCastNode(token, valueType, value));
    }

    void parenCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << '(';
      valueType.print(pout);
      pout << ") ";
      value.print(pout);
    }

    void parenCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      valueType.print(pout);
      std::cerr << "] (parenCast)\n";
      value.childDebugPrint(prefix);
    }

    constCastNode::constCastNode(token_t *token_,
                                 type_t &valueType_,
                                 exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    constCastNode::constCastNode(const constCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value.clone()) {}

    constCastNode::~constCastNode() {
      delete &value;
    }

    int constCastNode::type() const {
      return exprNodeType::constCast;
    }

    exprNode& constCastNode::clone() const {
      return *(new constCastNode(token, valueType, value));
    }

    void constCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "const_cast<";
      valueType.print(pout);
      pout << ">(";
      value.print(pout);
      pout << ')';
    }

    void constCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      valueType.print(pout);
      std::cerr << "] (constCast)\n";
      value.childDebugPrint(prefix);
    }

    staticCastNode::staticCastNode(token_t *token_,
                                   type_t &valueType_,
                                   exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    staticCastNode::staticCastNode(const staticCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value.clone()) {}

    staticCastNode::~staticCastNode() {
      delete &value;
    }

    int staticCastNode::type() const {
      return exprNodeType::staticCast;
    }

    exprNode& staticCastNode::clone() const {
      return *(new staticCastNode(token, valueType, value));
    }

    void staticCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "static_cast<";
      valueType.print(pout);
      pout << ">(";
      value.print(pout);
      pout << ')';
    }

    void staticCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      valueType.print(pout);
      std::cerr << "] (staticCast)\n";
      value.childDebugPrint(prefix);
    }

    reinterpretCastNode::reinterpretCastNode(token_t *token_,
                                             type_t &valueType_,
                                             exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    reinterpretCastNode::reinterpretCastNode(const reinterpretCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value.clone()) {}

    reinterpretCastNode::~reinterpretCastNode() {
      delete &value;
    }

    int reinterpretCastNode::type() const {
      return exprNodeType::reinterpretCast;
    }

    exprNode& reinterpretCastNode::clone() const {
      return *(new reinterpretCastNode(token, valueType, value));
    }

    void reinterpretCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "reinterpret_cast<";
      valueType.print(pout);
      pout << ">(";
      value.print(pout);
      pout << ')';
    }

    void reinterpretCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      valueType.print(pout);
      std::cerr << "] (reinterpretCast)\n";
      value.childDebugPrint(prefix);
    }

    dynamicCastNode::dynamicCastNode(token_t *token_,
                                     type_t &valueType_,
                                     exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    dynamicCastNode::dynamicCastNode(const dynamicCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value.clone()) {}

    dynamicCastNode::~dynamicCastNode() {
      delete &value;
    }

    int dynamicCastNode::type() const {
      return exprNodeType::dynamicCast;
    }

    exprNode& dynamicCastNode::clone() const {
      return *(new dynamicCastNode(token, valueType, value));
    }

    void dynamicCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "dynamic_cast<";
      valueType.print(pout);
      pout << ">(";
      value.print(pout);
      pout << ')';
    }

    void dynamicCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      valueType.print(pout);
      std::cerr << "] (dynamicCast)\n";
      value.childDebugPrint(prefix);
    }
    //==================================

    //---[ Misc ]-----------------------
    pairNode::pairNode(operatorToken &opToken,
                       exprNode &value_) :
      exprNode(&opToken),
      op(opToken.op),
      value(value_.clone()) {}

    pairNode::pairNode(const pairNode &node) :
      exprNode(node.token),
      op(node.op),
      value(node.value.clone()) {}

    pairNode::~pairNode() {
      delete &value;
    }

    int pairNode::type() const {
      return exprNodeType::pair;
    }

    exprNode& pairNode::clone() const {
      return *(new pairNode(token->to<operatorToken>(),
                            value.clone()));
    }

    bool pairNode::canEvaluate() const {
      token->printError("[Waldo] (pairNode) Unsure how you got here...");
      return false;
    }

    primitive pairNode::evaluate() const {
      token->printError("[Waldo] (pairNode) Unsure how you got here...");
      return primitive();
    }

    void pairNode::print(printer &pout) const {
      token->printError("[Waldo] (pairNode) Unsure how you got here...");
    }

    void pairNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      op.print(pout);
      std::cerr << "] (pairNode)\n";
      value.childDebugPrint(prefix);
    }

    parenthesesNode::parenthesesNode(token_t *token_,
                                     exprNode &value_) :
      exprNode(token_),
      value(value_.clone()) {}

    parenthesesNode::parenthesesNode(const parenthesesNode &node) :
      exprNode(node.token),
      value(node.value.clone()) {}

    parenthesesNode::~parenthesesNode() {
      delete &value;
    }

    int parenthesesNode::type() const {
      return exprNodeType::parentheses;
    }

    exprNode& parenthesesNode::clone() const {
      return *(new parenthesesNode(token, value));
    }

    bool parenthesesNode::canEvaluate() const {
      return value.canEvaluate();
    }

    primitive parenthesesNode::evaluate() const {
      return value.evaluate();
    }

    void parenthesesNode::print(printer &pout) const {
      pout << '(';
      value.print(pout);
      pout << ')';
    }

    void parenthesesNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[()] (parentheses)\n";
      value.childDebugPrint(prefix);
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

    int tupleNode::type() const {
      return exprNodeType::tuple;
    }

    exprNode& tupleNode::clone() const {
      return *(new tupleNode(token, args));
    }

    void tupleNode::print(printer &pout) const {
      pout << '{';
      const int argCount = (int) args.size();
      for (int i = 0; i < argCount; ++i) {
        if (i) {
          pout << ", ";
        }
        args[i]->print(pout);
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
                               exprNode &value_,
                               exprNode &blocks_,
                               exprNode &threads_) :
      exprNode(token_),
      value(value_.clone()),
      blocks(blocks_.clone()),
      threads(threads_.clone()) {}

    cudaCallNode::cudaCallNode(const cudaCallNode &node) :
      exprNode(node.token),
      value(node.value.clone()),
      blocks(node.blocks.clone()),
      threads(node.threads.clone()) {}

    cudaCallNode::~cudaCallNode() {
      delete &value;
      delete &blocks;
      delete &threads;
    }

    int cudaCallNode::type() const {
      return exprNodeType::cudaCall;
    }

    exprNode& cudaCallNode::clone() const {
      return *(new cudaCallNode(token,
                                value,
                                blocks,
                                threads));
    }

    void cudaCallNode::print(printer &pout) const {
      value.print(pout);
      pout << "<<<";
      blocks.print(pout);
      pout << ", ";
      threads.print(pout);
      pout << ">>>";
    }

    void cudaCallNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[<<<...>>>";
      std::cerr << "] (cudaCall)\n";
      value.childDebugPrint(prefix);
      blocks.childDebugPrint(prefix);
      threads.childDebugPrint(prefix);
    }
    //==================================
  }
}
