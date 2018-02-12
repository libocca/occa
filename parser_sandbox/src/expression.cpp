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
    exprNode::exprNode(token_t *token_) :
      token(token_) {}

    exprNode::~exprNode() {}

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
    }

    void exprNode::childDebugPrint(const std::string &prefix) const {
      debugPrint(prefix + "|   ");
    }

    // Using Shunting-Yard algorithm
    exprNode* exprNode::load(const tokenVector &tokens) {
      // TODO: Delete tokens
      // TODO: Ternary operator
      exprNodeStack output;
      operatorStack operators;

      const int outputTokenType = (tokenType::identifier |
                                   tokenType::primitive  |
                                   tokenType::char_      |
                                   tokenType::string);

      const int count = (int) tokens.size();
      for (int i = 0; i < count; ++i) {
        token_t *token = tokens[i];
        if (!token) {
          continue;
        }
        const int tokenType = token->type();
        bool success = true;
        if (tokenType & outputTokenType) {
          pushOutputNode(token,
                         output);
        }
        else if (tokenType & tokenType::op) {
          operatorToken &opToken = token->to<operatorToken>();

          if (opToken.op.opType & operatorType::pairStart) {
            operators.push(&opToken);
          }
          else if (opToken.op.opType & operatorType::pairEnd) {
            success = closePair(opToken,
                                     output,
                                operators);
          }
          else {
            success = applyFasterOperators(opToken,
                                           output,
                                           operators);
            operators.push(&opToken);
          }
        }
        if (!success) {
          return NULL;
        }
      }

      // Finish applying operators
      while (operators.size()) {
        operatorToken &opToken = *(operators.top());
        bool success = applyOperator(opToken,
                                     output,
                                     operators);
        if (!success) {
          return NULL;
        }
        operators.pop();
      }

      // Make sure we only have 1 root node
      const int outputNodes = (int) output.size();
      if (!outputNodes) {
        return NULL;
      }
      if (outputNodes > 1) {
        output.pop();
        output.top()->token->printError("Unable to form an expression");
        return NULL;
      }

      // Return the root node
      return output.top();
    }

    void exprNode::pushOutputNode(token_t *token,
                                  exprNodeStack &output) {
      const int tokenType = token->type();
      if (tokenType & tokenType::identifier) {
        identifierToken &t = token->to<identifierToken>();
        output.push(new identifierNode(token, t.value));
      }
      else if (tokenType & tokenType::primitive) {
        primitiveToken &t = token->to<primitiveToken>();
        output.push(new primitiveNode(token, t.value));
      }
      else if (tokenType & tokenType::char_) {
        // TODO: Handle char udfs here
        charToken &t = token->to<charToken>();
        output.push(new charNode(token, t.value));
      }
      else if (tokenType & tokenType::string) {
        // TODO: Handle string udfs here
        stringToken &t = token->to<stringToken>();
        output.push(new stringNode(token, t.value));
      }
    }

    bool exprNode::closePair(operatorToken &opToken,
                             exprNodeStack &output,
                             operatorStack &operators) {
      const opType_t opType = opToken.op.opType;
      operatorToken *errorToken = &opToken;

      while (operators.size()) {
        operatorToken &nextOpToken = *(operators.top());
        const opType_t nextOpType = nextOpToken.op.opType;
        operators.pop();

        if (nextOpType & operatorType::pairStart) {
          if (opType == (nextOpType << 1)) {
            return applyOperator(opToken,
                                 output,
                                 operators);
          }
          errorToken = &nextOpToken;
          break;
        }

        bool success = applyOperator(nextOpToken,
                                     output,
                                     operators);
        if (!success) {
          return false;
        }
      }
      // Found a pairStart that doesn't match
      const opType_t errorOpType = errorToken->op.opType;
      std::stringstream ss;
      ss << "Could not find an opening ";
      if (errorOpType & operatorType::braceStart) {
        ss << '{';
      }
      else if (errorOpType & operatorType::bracketStart) {
        ss << '[';
      }
      else if (errorOpType & operatorType::parenthesesStart) {
        ss << '(';
      }
      errorToken->printError(ss.str());
      return false;
    }

    bool exprNode::applyFasterOperators(operatorToken &opToken,
                                        exprNodeStack &output,
                                        operatorStack &operators) {
      const operator_t &op = opToken.op;
      while (operators.size()) {
        const operator_t &prevOp = operators.top()->op;

        if (prevOp.opType & operatorType::pairStart) {
          break;
        }

        if ((op.precedence > prevOp.precedence) ||
            ((op.precedence == prevOp.precedence) &&
             op::associativity[prevOp.precedence] == op::leftAssociative)) {

          bool success = applyOperator(*operators.top(),
                                       output,
                                       operators);
          if (!success) {
            return false;
          }
          operators.pop();
          continue;
        }

        break;
      }
      return true;
    }

    bool exprNode::applyOperator(operatorToken &opToken,
                                 exprNodeStack &output,
                                 operatorStack &operators) {
      const operator_t &op = opToken.op;
      const opType_t opType = op.opType;
      const int outputCount = (int) output.size();

      if (!outputCount) {
        return false;
      }

      exprNode &value = *(output.top());
      output.pop();

      if (opType & operatorType::binary) {
        if (outputCount < 2) {
          return false;
        }
        exprNode &left = *(output.top());
        output.pop();
        output.push(new binaryOpNode(&opToken,
                                     (const binaryOperator_t&) op,
                                     left,
                                     value));
      }
      else if (opType & operatorType::leftUnary) {
        output.push(new leftUnaryOpNode(&opToken,
                                        (const unaryOperator_t&) op,
                                        value));
      }
      else if (opType & operatorType::rightUnary) {
        output.push(new rightUnaryOpNode(&opToken,
                                         (const unaryOperator_t&) op,
                                         value));
      }
      else if (opType & operatorType::pair) {
        output.push(new parenthesesNode(&opToken,
                                        value));
      } else {
        return false;
      }

      return true;
    }

    //---[ Empty ]----------------------
    emptyNode::emptyNode() :
      exprNode() {}

    emptyNode::~emptyNode() {}

    int emptyNode::nodeType() const {
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
    primitiveNode::primitiveNode(primitive value_) :
      value(value_) {}

    primitiveNode::primitiveNode(token_t *token_,
                                 primitive value_) :
      exprNode(token_),
      value(value_) {}

    primitiveNode::primitiveNode(const primitiveNode &node) :
      value(node.value) {}

    primitiveNode::~primitiveNode() {}

    int primitiveNode::nodeType() const {
      return exprNodeType::primitive;
    }

    exprNode& primitiveNode::clone() const {
      return *(new primitiveNode(value));
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
    charNode::charNode(const std::string &value_) :
      value(value_) {}

    charNode::charNode(token_t *token_,
                       const std::string &value_) :
      exprNode(token_),
      value(value_) {}

    charNode::charNode(const charNode &node) :
      value(node.value) {}

    charNode::~charNode() {}

    int charNode::nodeType() const {
      return exprNodeType::char_;
    }

    exprNode& charNode::clone() const {
      return *(new charNode(value));
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
    stringNode::stringNode(const std::string &value_) :
      value(value_) {}

    stringNode::stringNode(token_t *token_,
                           const std::string &value_) :
      exprNode(token_),
      value(value_) {}

    stringNode::stringNode(const stringNode &node) :
      value(node.value) {}

    stringNode::~stringNode() {}

    int stringNode::nodeType() const {
      return exprNodeType::string;
    }

    exprNode& stringNode::clone() const {
      return *(new stringNode(value));
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
    identifierNode::identifierNode(const std::string &value_) :
      value(value_) {}

    identifierNode::identifierNode(token_t *token_,
                                   const std::string &value_) :
      exprNode(token_),
      value(value_) {}

    identifierNode::identifierNode(const identifierNode &node) :
      value(node.value) {}

    identifierNode::~identifierNode() {}

    int identifierNode::nodeType() const {
      return exprNodeType::identifier;
    }

    exprNode& identifierNode::clone() const {
      return *(new identifierNode(value));
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
    variableNode::variableNode(variable &value_) :
      value(value_) {}

    variableNode::variableNode(token_t *token_,
                               variable &value_) :
      exprNode(token_),
      value(value_) {}

    variableNode::variableNode(const variableNode &node) :
      value(node.value) {}

    variableNode::~variableNode() {}

    int variableNode::nodeType() const {
      return exprNodeType::variable;
    }

    exprNode& variableNode::clone() const {
      return *(new variableNode(value));
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
    leftUnaryOpNode::leftUnaryOpNode(const unaryOperator_t &op_,
                                     exprNode &value_) :
      op(op_),
      value(value_.clone()) {}

    leftUnaryOpNode::leftUnaryOpNode(token_t *token_,
                                     const unaryOperator_t &op_,
                                     exprNode &value_) :
      exprNode(token_),
      op(op_),
      value(value_.clone()) {}

    leftUnaryOpNode::leftUnaryOpNode(const leftUnaryOpNode &node) :
      op(node.op),
      value(node.value.clone()) {}

    leftUnaryOpNode::~leftUnaryOpNode() {
      delete &value;
    }

    int leftUnaryOpNode::nodeType() const {
      return exprNodeType::leftUnary;
    }

    opType_t leftUnaryOpNode::opnodeType() const {
      return op.opType;
    }

    exprNode& leftUnaryOpNode::clone() const {
      return *(new leftUnaryOpNode(op, value));
    }

    bool leftUnaryOpNode::canEvaluate() const {
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

    rightUnaryOpNode::rightUnaryOpNode(const unaryOperator_t &op_,
                                       exprNode &value_) :
      op(op_),
      value(value_.clone()) {}

    rightUnaryOpNode::rightUnaryOpNode(token_t *token_,
                                       const unaryOperator_t &op_,
                                       exprNode &value_) :
      exprNode(token_),
      op(op_),
      value(value_.clone()) {}

    rightUnaryOpNode::rightUnaryOpNode(const rightUnaryOpNode &node) :
      op(node.op),
      value(node.value.clone()) {}

    rightUnaryOpNode::~rightUnaryOpNode() {
      delete &value;
    }

    int rightUnaryOpNode::nodeType() const {
      return exprNodeType::rightUnary;
    }

    opType_t rightUnaryOpNode::opnodeType() const {
      return op.opType;
    }

    exprNode& rightUnaryOpNode::clone() const {
      return *(new rightUnaryOpNode(op, value));
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

    binaryOpNode::binaryOpNode(const binaryOperator_t &op_,
                               exprNode &leftValue_,
                               exprNode &rightValue_) :
      op(op_),
      leftValue(leftValue_.clone()),
      rightValue(rightValue_.clone()) {}

    binaryOpNode::binaryOpNode(token_t *token_,
                               const binaryOperator_t &op_,
                               exprNode &leftValue_,
                               exprNode &rightValue_) :
      exprNode(token_),
      op(op_),
      leftValue(leftValue_.clone()),
      rightValue(rightValue_.clone()) {}

    binaryOpNode::binaryOpNode(const binaryOpNode &node) :
      op(node.op),
      leftValue(node.leftValue.clone()),
      rightValue(node.rightValue.clone()) {}

    binaryOpNode::~binaryOpNode() {
      delete &leftValue;
      delete &rightValue;
    }

    int binaryOpNode::nodeType() const {
      return exprNodeType::binary;
    }

    opType_t binaryOpNode::opnodeType() const {
      return op.opType;
    }

    exprNode& binaryOpNode::clone() const {
      return *(new binaryOpNode(op, leftValue, rightValue));
    }

    bool binaryOpNode::canEvaluate() const {
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

    ternaryOpNode::ternaryOpNode(exprNode &checkValue_,
                                 exprNode &trueValue_,
                                 exprNode &falseValue_) :
      checkValue(checkValue_.clone()),
      trueValue(trueValue_.clone()),
      falseValue(falseValue_.clone()) {}

    ternaryOpNode::ternaryOpNode(token_t *token_,
                                 exprNode &checkValue_,
                                 exprNode &trueValue_,
                                 exprNode &falseValue_) :
      exprNode(token_),
      checkValue(checkValue_.clone()),
      trueValue(trueValue_.clone()),
      falseValue(falseValue_.clone()) {}

    ternaryOpNode::ternaryOpNode(const ternaryOpNode &node) :
      checkValue(node.checkValue.clone()),
      trueValue(node.trueValue.clone()),
      falseValue(node.falseValue.clone()) {}

    ternaryOpNode::~ternaryOpNode() {
      delete &checkValue;
      delete &trueValue;
      delete &falseValue;
    }

    int ternaryOpNode::nodeType() const {
      return exprNodeType::ternary;
    }

    opType_t ternaryOpNode::opnodeType() const {
      return operatorType::ternary;
    }

    exprNode& ternaryOpNode::clone() const {
      return *(new ternaryOpNode(checkValue,
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
    subscriptNode::subscriptNode(exprNode &value_,
                                 exprNode &index_) :
      value(value_.clone()),
      index(index_.clone()) {}

    subscriptNode::subscriptNode(token_t *token_,
                                 exprNode &value_,
                                 exprNode &index_) :
      exprNode(token_),
      value(value_.clone()),
      index(index_.clone()) {}

    subscriptNode::subscriptNode(const subscriptNode &node) :
      value(node.value.clone()),
      index(node.index.clone()) {}

    subscriptNode::~subscriptNode() {
      delete &value;
      delete &index;
    }

    int subscriptNode::nodeType() const {
      return exprNodeType::subscript;
    }

    exprNode& subscriptNode::clone() const {
      return *(new subscriptNode(value, index));
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

    callNode::callNode(exprNode &value_,
                       exprNodeVector args_) :
      value(value_.clone()) {
      const int argCount = (int) args_.size();
      for (int i = 0; i < argCount; ++i) {
        args.push_back(&(args_[i]->clone()));
      }
    }

    callNode::callNode(token_t *token_,
                       exprNode &value_,
                       exprNodeVector args_) :
      exprNode(token_),
      value(value_.clone()) {
      const int argCount = (int) args_.size();
      for (int i = 0; i < argCount; ++i) {
        args.push_back(&(args_[i]->clone()));
      }
    }

    callNode::callNode(const callNode &node) :
      value(node.value.clone()) {
      const int argCount = (int) node.args.size();
      for (int i = 0; i < argCount; ++i) {
        args.push_back(&(node.args[i]->clone()));
      }
    }

    callNode::~callNode() {
      delete &value;
      const int argCount = (int) args.size();
      for (int i = 0; i < argCount; ++i) {
        delete args[i];
      }
      args.clear();
    }

    int callNode::nodeType() const {
      return exprNodeType::call;
    }

    exprNode& callNode::clone() const {
      return *(new callNode(value, args));
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

    newNode::newNode(type_t &type_,
                     exprNode &value_) :
      type(type_),
      value(value_.clone()),
      size(*(new emptyNode())) {}

    newNode::newNode(type_t &type_,
                     exprNode &value_,
                     exprNode &size_) :
      type(type_),
      value(value_.clone()),
      size(size_.clone()) {}

    newNode::newNode(token_t *token_,
                     type_t &type_,
                     exprNode &value_,
                     exprNode &size_) :
      exprNode(token_),
      type(type_),
      value(value_.clone()),
      size(size_.clone()) {}

    newNode::newNode(const newNode &node) :
      type(node.type),
      value(node.value.clone()),
      size(node.size.clone()) {}

    newNode::~newNode() {
      delete &value;
      delete &size;
    }

    int newNode::nodeType() const {
      return exprNodeType::new_;
    }

    exprNode& newNode::clone() const {
      return *(new newNode(type, value, size));
    }

    void newNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "new ";
      type.print(pout);
      value.print(pout);
      if (size.nodeType() != exprNodeType::empty) {
        pout << '[';
        size.print(pout);
        pout << ']';
      }
    }

    void newNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      type.print(pout);
      std::cerr << "] (new)\n";
      value.childDebugPrint(prefix);
      size.childDebugPrint(prefix);
    }

    deleteNode::deleteNode(exprNode &value_,
                           const bool isArray_) :
      value(value_.clone()),
      isArray(isArray_) {}

    deleteNode::deleteNode(token_t *token_,
                           exprNode &value_,
                           const bool isArray_) :
      exprNode(token_),
      value(value_.clone()),
      isArray(isArray_) {}

    deleteNode::deleteNode(const deleteNode &node) :
      value(node.value.clone()),
      isArray(node.isArray) {}

    deleteNode::~deleteNode() {
      delete &value;
    }

    int deleteNode::nodeType() const {
      return exprNodeType::delete_;
    }

    exprNode& deleteNode::clone() const {
      return *(new deleteNode(value, isArray));
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

    throwNode::throwNode(exprNode &value_) :
      value(value_.clone()) {}

    throwNode::throwNode(token_t *token_,
                         exprNode &value_) :
      exprNode(token_),
      value(value_.clone()) {}

    throwNode::throwNode(const throwNode &node) :
      value(node.value.clone()) {}

    throwNode::~throwNode() {
      delete &value;
    }

    int throwNode::nodeType() const {
      return exprNodeType::throw_;
    }

    exprNode& throwNode::clone() const {
      return *(new throwNode(value));
    }

    void throwNode::print(printer &pout) const {
      pout << "throw";
      if (value.nodeType() != exprNodeType::empty) {
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
    sizeofNode::sizeofNode(exprNode &value_) :
      value(value_.clone()) {}

    sizeofNode::sizeofNode(token_t *token_,
                           exprNode &value_) :
      exprNode(token_),
      value(value_.clone()) {}

    sizeofNode::sizeofNode(const sizeofNode &node) :
      value(node.value.clone()) {}

    sizeofNode::~sizeofNode() {
      delete &value;
    }

    int sizeofNode::nodeType() const {
      return exprNodeType::sizeof_;
    }

    exprNode& sizeofNode::clone() const {
      return *(new sizeofNode(value));
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

    funcCastNode::funcCastNode(type_t &type_,
                               exprNode &value_) :
      type(type_),
      value(value_.clone()) {}

    funcCastNode::funcCastNode(token_t *token_,
                               type_t &type_,
                               exprNode &value_) :
      exprNode(token_),
      type(type_),
      value(value_.clone()) {}

    funcCastNode::funcCastNode(const funcCastNode &node) :
      type(node.type),
      value(node.value.clone()) {}

    funcCastNode::~funcCastNode() {
      delete &value;
    }

    int funcCastNode::nodeType() const {
      return exprNodeType::funcCast;
    }

    exprNode& funcCastNode::clone() const {
      return *(new funcCastNode(type, value));
    }

    void funcCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      type.print(pout);
      pout << '(';
      value.print(pout);
      pout << ')';
    }

    void funcCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      type.print(pout);
      std::cerr << "] (funcCast)\n";
      value.childDebugPrint(prefix);
    }

    parenCastNode::parenCastNode(type_t &type_,
                                 exprNode &value_) :
      type(type_),
      value(value_.clone()) {}

    parenCastNode::parenCastNode(token_t *token_,
                                 type_t &type_,
                                 exprNode &value_) :
      exprNode(token_),
      type(type_),
      value(value_.clone()) {}

    parenCastNode::parenCastNode(const parenCastNode &node) :
      type(node.type),
      value(node.value.clone()) {}

    parenCastNode::~parenCastNode() {
      delete &value;
    }

    int parenCastNode::nodeType() const {
      return exprNodeType::parenCast;
    }

    exprNode& parenCastNode::clone() const {
      return *(new parenCastNode(type, value));
    }

    void parenCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << '(';
      type.print(pout);
      pout << ") ";
      value.print(pout);
    }

    void parenCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      type.print(pout);
      std::cerr << "] (parenCast)\n";
      value.childDebugPrint(prefix);
    }

    constCastNode::constCastNode(type_t &type_,
                                 exprNode &value_) :
      type(type_),
      value(value_.clone()) {}

    constCastNode::constCastNode(token_t *token_,
                                 type_t &type_,
                                 exprNode &value_) :
      exprNode(token_),
      type(type_),
      value(value_.clone()) {}

    constCastNode::constCastNode(const constCastNode &node) :
      type(node.type),
      value(node.value.clone()) {}

    constCastNode::~constCastNode() {
      delete &value;
    }

    int constCastNode::nodeType() const {
      return exprNodeType::constCast;
    }

    exprNode& constCastNode::clone() const {
      return *(new constCastNode(type, value));
    }

    void constCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "const_cast<";
      type.print(pout);
      pout << ">(";
      value.print(pout);
      pout << ')';
    }

    void constCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      type.print(pout);
      std::cerr << "] (constCast)\n";
      value.childDebugPrint(prefix);
    }

    staticCastNode::staticCastNode(type_t &type_,
                                   exprNode &value_) :
      type(type_),
      value(value_.clone()) {}

    staticCastNode::staticCastNode(token_t *token_,
                                   type_t &type_,
                                   exprNode &value_) :
      exprNode(token_),
      type(type_),
      value(value_.clone()) {}

    staticCastNode::staticCastNode(const staticCastNode &node) :
      type(node.type),
      value(node.value.clone()) {}

    staticCastNode::~staticCastNode() {
      delete &value;
    }

    int staticCastNode::nodeType() const {
      return exprNodeType::staticCast;
    }

    exprNode& staticCastNode::clone() const {
      return *(new staticCastNode(type, value));
    }

    void staticCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "static_cast<";
      type.print(pout);
      pout << ">(";
      value.print(pout);
      pout << ')';
    }

    void staticCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      type.print(pout);
      std::cerr << "] (staticCast)\n";
      value.childDebugPrint(prefix);
    }

    reinterpretCastNode::reinterpretCastNode(type_t &type_,
                                             exprNode &value_) :
      type(type_),
      value(value_.clone()) {}

    reinterpretCastNode::reinterpretCastNode(token_t *token_,
                                             type_t &type_,
                                             exprNode &value_) :
      exprNode(token_),
      type(type_),
      value(value_.clone()) {}

    reinterpretCastNode::reinterpretCastNode(const reinterpretCastNode &node) :
      type(node.type),
      value(node.value.clone()) {}

    reinterpretCastNode::~reinterpretCastNode() {
      delete &value;
    }

    int reinterpretCastNode::nodeType() const {
      return exprNodeType::reinterpretCast;
    }

    exprNode& reinterpretCastNode::clone() const {
      return *(new reinterpretCastNode(type, value));
    }

    void reinterpretCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "reinterpret_cast<";
      type.print(pout);
      pout << ">(";
      value.print(pout);
      pout << ')';
    }

    void reinterpretCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      type.print(pout);
      std::cerr << "] (reinterpretCast)\n";
      value.childDebugPrint(prefix);
    }

    dynamicCastNode::dynamicCastNode(type_t &type_,
                                     exprNode &value_) :
      type(type_),
      value(value_.clone()) {}

    dynamicCastNode::dynamicCastNode(token_t *token_,
                                     type_t &type_,
                                     exprNode &value_) :
      exprNode(token_),
      type(type_),
      value(value_.clone()) {}

    dynamicCastNode::dynamicCastNode(const dynamicCastNode &node) :
      type(node.type),
      value(node.value.clone()) {}

    dynamicCastNode::~dynamicCastNode() {
      delete &value;
    }

    int dynamicCastNode::nodeType() const {
      return exprNodeType::dynamicCast;
    }

    exprNode& dynamicCastNode::clone() const {
      return *(new dynamicCastNode(type, value));
    }

    void dynamicCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "dynamic_cast<";
      type.print(pout);
      pout << ">(";
      value.print(pout);
      pout << ')';
    }

    void dynamicCastNode::debugPrint(const std::string &prefix) const {
      printer pout(std::cerr);
      std::cerr << prefix << "|\n"
                << prefix << "|---[";
      type.print(pout);
      std::cerr << "] (dynamicCast)\n";
      value.childDebugPrint(prefix);
    }
    //==================================

    //---[ Misc ]-----------------------
    parenthesesNode::parenthesesNode(exprNode &value_) :
      value(value_.clone()) {}

    parenthesesNode::parenthesesNode(token_t *token_,
                                     exprNode &value_) :
      exprNode(token_),
      value(value_.clone()) {}

    parenthesesNode::parenthesesNode(const parenthesesNode &node) :
      value(node.value.clone()) {}

    parenthesesNode::~parenthesesNode() {
      delete &value;
    }

    int parenthesesNode::nodeType() const {
      return exprNodeType::parentheses;
    }

    exprNode& parenthesesNode::clone() const {
      return *(new parenthesesNode(value));
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
    //==================================

    //---[ Extensions ]-----------------
    cudaCallNode::cudaCallNode(exprNode &blocks_,
                               exprNode &threads_) :
      blocks(blocks_.clone()),
      threads(threads_.clone()) {}

    cudaCallNode::cudaCallNode(token_t *token_,
                               exprNode &blocks_,
                               exprNode &threads_) :
      exprNode(token_),
      blocks(blocks_.clone()),
      threads(threads_.clone()) {}

    cudaCallNode::cudaCallNode(const cudaCallNode &node) :
      blocks(node.blocks.clone()),
      threads(node.threads.clone()) {}

    cudaCallNode::~cudaCallNode() {
      delete &blocks;
      delete &threads;
    }

    int cudaCallNode::nodeType() const {
      return exprNodeType::cudaCall;
    }

    exprNode& cudaCallNode::clone() const {
      return *(new cudaCallNode(blocks, threads));
    }

    void cudaCallNode::print(printer &pout) const {
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
      blocks.childDebugPrint(prefix);
      threads.childDebugPrint(prefix);
    }
    //==================================
  }
}
