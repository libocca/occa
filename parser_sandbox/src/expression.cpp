#include "expression.hpp"

namespace occa {
  namespace lang {
    exprNode::~exprNode() {}

    std::string exprNode::toString() const {
      std::stringstream ss;
      printer_t pout(ss);
      print(pout);
      return ss.str();
    }

    void exprNode::debugPrint() const {
      std::cout << toString();
    }

    //---[ Empty ]----------------------
    emptyNode::emptyNode() {}
    emptyNode::~emptyNode() {}

    int emptyNode::nodeType() const {
      return exprNodeType::empty;
    }

    exprNode& emptyNode::clone() const {
      return *(new emptyNode());
    }

    void emptyNode::print(printer_t &pout) const {}

    const emptyNode noExprNode;
    //==================================

    //---[ Values ]---------------------
    primitiveNode::primitiveNode(primitive value_) :
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

    void primitiveNode::print(printer_t &pout) const {
      pout << (std::string) value;
    }

    variableNode::variableNode(variable_t &value_) :
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

    void variableNode::print(printer_t &pout) const {
      value.print(pout);
    }
    //==================================

    //---[ Operators ]------------------
    opNode::opNode(const operator_t &op_) :
      op(op_) {}

    int opNode::opnodeType() const {
      return op.optype;
    }

    leftUnaryOpNode::leftUnaryOpNode(const operator_t &op_,
                                     exprNode &value_) :
      opNode(op_),
      value(value_.clone()) {}

    leftUnaryOpNode::leftUnaryOpNode(const leftUnaryOpNode &node) :
      opNode(node.op),
      value(node.value.clone()) {}

    leftUnaryOpNode::~leftUnaryOpNode() {
      delete &value;
    }

    int leftUnaryOpNode::nodeType() const {
      return exprNodeType::leftUnary;
    }

    exprNode& leftUnaryOpNode::clone() const {
      return *(new leftUnaryOpNode(op, value));
    }

    void leftUnaryOpNode::print(printer_t &pout) const {
      op.print(pout);
      value.print(pout);
    }

    rightUnaryOpNode::rightUnaryOpNode(const operator_t &op_,
                                       exprNode &value_) :
      opNode(op_),
      value(value_.clone()) {}

    rightUnaryOpNode::rightUnaryOpNode(const rightUnaryOpNode &node) :
      opNode(node.op),
      value(node.value.clone()) {}

    rightUnaryOpNode::~rightUnaryOpNode() {
      delete &value;
    }

    int rightUnaryOpNode::nodeType() const {
      return exprNodeType::rightUnary;
    }

    exprNode& rightUnaryOpNode::clone() const {
      return *(new rightUnaryOpNode(op, value));
    }

    void rightUnaryOpNode::print(printer_t &pout) const {
      value.print(pout);
      op.print(pout);
    }

    binaryOpNode::binaryOpNode(const operator_t &op_,
                               exprNode &leftValue_,
                               exprNode &rightValue_) :
      opNode(op_),
      leftValue(leftValue_.clone()),
      rightValue(rightValue_.clone()) {}

    binaryOpNode::binaryOpNode(const binaryOpNode &node) :
      opNode(node.op),
      leftValue(node.leftValue.clone()),
      rightValue(node.rightValue.clone()) {}

    binaryOpNode::~binaryOpNode() {
      delete &leftValue;
      delete &rightValue;
    }

    int binaryOpNode::nodeType() const {
      return exprNodeType::binary;
    }

    exprNode& binaryOpNode::clone() const {
      return *(new binaryOpNode(op, leftValue, rightValue));
    }

    void binaryOpNode::print(printer_t &pout) const {
      leftValue.print(pout);
      pout << ' ';
      op.print(pout);
      pout << ' ';
      rightValue.print(pout);
    }

    ternaryOpNode::ternaryOpNode(const operator_t &op_,
                                 exprNode &checkValue_,
                                 exprNode &trueValue_,
                                 exprNode &falseValue_) :
      opNode(op_),
      checkValue(checkValue_.clone()),
      trueValue(trueValue_.clone()),
      falseValue(falseValue_.clone()) {}

    ternaryOpNode::ternaryOpNode(const ternaryOpNode &node) :
      opNode(node.op),
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

    exprNode& ternaryOpNode::clone() const {
      return *(new ternaryOpNode(op, checkValue, trueValue, falseValue));
    }

    void ternaryOpNode::print(printer_t &pout) const {
      checkValue.print(pout);
      pout << " ? ";
      trueValue.print(pout);
      pout << " : ";
      falseValue.print(pout);
    }
    //==================================

    //---[ Pseudo Operators ]-----------
    subscriptNode::subscriptNode(exprNode &value_,
                                 exprNode &index_) :
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

    void subscriptNode::print(printer_t &pout) const {
      value.print(pout);
      pout << '[';
      index.print(pout);
      pout << ']';
    }

    callNode::callNode(exprNode &value_,
                       exprNodeVector_t args_) :
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

    void callNode::print(printer_t &pout) const {
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

    void newNode::print(printer_t &pout) const {
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

    deleteNode::deleteNode(exprNode &value_,
                           const bool isArray_) :
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

    void deleteNode::print(printer_t &pout) const {
      pout << "delete ";
      if (isArray) {
        pout << "[] ";
      }
      value.print(pout);
    }

    throwNode::throwNode(exprNode &value_) :
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

    void throwNode::print(printer_t &pout) const {
      pout << "throw";
      if (value.nodeType() != exprNodeType::empty) {
        pout << ' ';
        value.print(pout);
      }
    }
    //==================================

    //---[ Builtins ]-------------------
    sizeofNode::sizeofNode(exprNode &value_) :
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

    void sizeofNode::print(printer_t &pout) const {
      pout << "sizeof(";
      value.print(pout);
      pout << ')';
    }

    funcCastNode::funcCastNode(type_t &type_,
                               exprNode &value_) :
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

    void funcCastNode::print(printer_t &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      type.print(pout);
      pout << '(';
      value.print(pout);
      pout << ')';
    }

    parenCastNode::parenCastNode(type_t &type_,
                                 exprNode &value_) :
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

    void parenCastNode::print(printer_t &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << '(';
      type.print(pout);
      pout << ") ";
      value.print(pout);
    }

    constCastNode::constCastNode(type_t &type_,
                                 exprNode &value_) :
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

    void constCastNode::print(printer_t &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "const_cast<";
      type.print(pout);
      pout << ">(";
      value.print(pout);
      pout << ')';
    }

    staticCastNode::staticCastNode(type_t &type_,
                                   exprNode &value_) :
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

    void staticCastNode::print(printer_t &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "static_cast<";
      type.print(pout);
      pout << ">(";
      value.print(pout);
      pout << ')';
    }

    reinterpretCastNode::reinterpretCastNode(type_t &type_,
                                             exprNode &value_) :
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

    void reinterpretCastNode::print(printer_t &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "reinterpret_cast<";
      type.print(pout);
      pout << ">(";
      value.print(pout);
      pout << ')';
    }

    dynamicCastNode::dynamicCastNode(type_t &type_,
                                     exprNode &value_) :
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

    void dynamicCastNode::print(printer_t &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "dynamic_cast<";
      type.print(pout);
      pout << ">(";
      value.print(pout);
      pout << ')';
    }
    //==================================

    //---[ Misc ]-----------------------
    parenthesesNode::parenthesesNode(exprNode &value_) :
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

    void parenthesesNode::print(printer_t &pout) const {
      pout << '(';
      value.print(pout);
      pout << ')';
    }
    //==================================

    //---[ Extensions ]-----------------
    cudaCallNode::cudaCallNode(exprNode &blocks_,
                               exprNode &threads_) :
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

    void cudaCallNode::print(printer_t &pout) const {
      pout << "<<<";
      blocks.print(pout);
      pout << ", ";
      threads.print(pout);
      pout << ">>>";
    }
    //==================================
  }
}
