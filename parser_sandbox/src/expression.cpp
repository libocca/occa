#include "expression.hpp"

namespace occa {
  namespace lang {
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

    int emptyNode::nodeType() const {
      return exprNodeType::empty;
    }

    void emptyNode::print(printer_t &pout) const {}

    const emptyNode noExprNode;
    //==================================

    //---[ Values ]---------------------
    primitiveNode::primitiveNode(primitive value_) :
      value(value_) {}

    primitiveNode::primitiveNode(const primitiveNode &node) :
      value(node.value) {}

    int primitiveNode::nodeType() const {
      return exprNodeType::primitive;
    }

    void primitiveNode::print(printer_t &pout) const {
      pout << (std::string) value;
    }

    variableNode::variableNode(variable_t &value_) :
      value(value_) {}

    variableNode::variableNode(const variableNode &node) :
      value(node.value) {}

    int variableNode::nodeType() const {
      return exprNodeType::variable;
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
      value(value_) {}

    leftUnaryOpNode::leftUnaryOpNode(const leftUnaryOpNode &node) :
      opNode(node.op),
      value(node.value) {}

    int leftUnaryOpNode::nodeType() const {
      return exprNodeType::leftUnary;
    }

    void leftUnaryOpNode::print(printer_t &pout) const {
      op.print(pout);
      value.print(pout);
    }

    rightUnaryOpNode::rightUnaryOpNode(const operator_t &op_,
                                       exprNode &value_) :
      opNode(op_),
      value(value_) {}

    rightUnaryOpNode::rightUnaryOpNode(const rightUnaryOpNode &node) :
      opNode(node.op),
      value(node.value) {}

    int rightUnaryOpNode::nodeType() const {
      return exprNodeType::rightUnary;
    }

    void rightUnaryOpNode::print(printer_t &pout) const {
      value.print(pout);
      op.print(pout);
    }

    binaryOpNode::binaryOpNode(const operator_t &op_,
                               exprNode &leftValue_,
                               exprNode &rightValue_) :
      opNode(op_),
      leftValue(leftValue_),
      rightValue(rightValue_) {}

    binaryOpNode::binaryOpNode(const binaryOpNode &node) :
      opNode(node.op),
      leftValue(node.leftValue),
      rightValue(node.rightValue) {}

    int binaryOpNode::nodeType() const {
      return exprNodeType::binary;
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
      checkValue(checkValue_),
      trueValue(trueValue_),
      falseValue(falseValue_) {}

    ternaryOpNode::ternaryOpNode(const ternaryOpNode &node) :
      opNode(node.op),
      checkValue(node.checkValue),
      trueValue(node.trueValue),
      falseValue(node.falseValue) {}

    int ternaryOpNode::nodeType() const {
      return exprNodeType::ternary;
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
      value(value_),
      index(index_) {}

    subscriptNode::subscriptNode(const subscriptNode &node) :
      value(node.value),
      index(node.index) {}

    int subscriptNode::nodeType() const {
      return exprNodeType::subscript;
    }

    void subscriptNode::print(printer_t &pout) const {
      value.print(pout);
      pout << '[';
      index.print(pout);
      pout << ']';
    }

    callNode::callNode(exprNode &value_,
                       exprNodeVector_t args_) :
      value(value_),
      args(args_) {}

    callNode::callNode(const callNode &node) :
      value(node.value),
      args(node.args) {}

    int callNode::nodeType() const {
      return exprNodeType::call;
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
      value(value_),
      size(const_cast<emptyNode&>(noExprNode)) {}

    newNode::newNode(type_t &type_,
                     exprNode &value_,
                     exprNode &size_) :
      type(type_),
      value(value_),
      size(size_) {}

    newNode::newNode(const newNode &node) :
      type(node.type),
      value(node.value),
      size(node.size) {}

    int newNode::nodeType() const {
      return exprNodeType::new_;
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
      value(value_),
      isArray(isArray_) {}

    deleteNode::deleteNode(const deleteNode &node) :
      value(node.value),
      isArray(node.isArray) {}

    int deleteNode::nodeType() const {
      return exprNodeType::delete_;
    }

    void deleteNode::print(printer_t &pout) const {
      pout << "delete ";
      if (isArray) {
        pout << "[] ";
      }
      value.print(pout);
    }

    throwNode::throwNode(exprNode &value_) :
      value(value_) {}

    throwNode::throwNode(const throwNode &node) :
      value(node.value) {}

    int throwNode::nodeType() const {
      return exprNodeType::throw_;
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
      value(value_) {}

    sizeofNode::sizeofNode(const sizeofNode &node) :
      value(node.value) {}

    int sizeofNode::nodeType() const {
      return exprNodeType::sizeof_;
    }

    void sizeofNode::print(printer_t &pout) const {
      pout << "sizeof(";
      value.print(pout);
      pout << ')';
    }

    funcCastNode::funcCastNode(type_t &type_,
                               exprNode &value_) :
      type(type_),
      value(value_) {}

    funcCastNode::funcCastNode(const funcCastNode &node) :
      type(node.type),
      value(node.value) {}

    int funcCastNode::nodeType() const {
      return exprNodeType::funcCast;
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
      value(value_) {}

    parenCastNode::parenCastNode(const parenCastNode &node) :
      type(node.type),
      value(node.value) {}

    int parenCastNode::nodeType() const {
      return exprNodeType::parenCast;
    }

    void parenCastNode::print(printer_t &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << '(';
      type.print(pout);
      pout << ") ";
      value.print(pout);
    }

    constCast::constCast(type_t &type_,
                         exprNode &value_) :
      type(type_),
      value(value_) {}

    constCast::constCast(const constCast &node) :
      type(node.type),
      value(node.value) {}

    int constCast::nodeType() const {
      return exprNodeType::constCast;
    }

    void constCast::print(printer_t &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "const_cast<";
      type.print(pout);
      pout << ">(";
      value.print(pout);
      pout << ')';
    }

    staticCast::staticCast(type_t &type_,
                           exprNode &value_) :
      type(type_),
      value(value_) {}

    staticCast::staticCast(const staticCast &node) :
      type(node.type),
      value(node.value) {}

    int staticCast::nodeType() const {
      return exprNodeType::staticCast;
    }

    void staticCast::print(printer_t &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "static_cast<";
      type.print(pout);
      pout << ">(";
      value.print(pout);
      pout << ')';
    }

    reinterpretCast::reinterpretCast(type_t &type_,
                                     exprNode &value_) :
      type(type_),
      value(value_) {}

    reinterpretCast::reinterpretCast(const reinterpretCast &node) :
      type(node.type),
      value(node.value) {}

    int reinterpretCast::nodeType() const {
      return exprNodeType::reinterpretCast;
    }

    void reinterpretCast::print(printer_t &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "reinterpret_cast<";
      type.print(pout);
      pout << ">(";
      value.print(pout);
      pout << ')';
    }

    dynamicCast::dynamicCast(type_t &type_,
                             exprNode &value_) :
      type(type_),
      value(value_) {}

    dynamicCast::dynamicCast(const dynamicCast &node) :
      type(node.type),
      value(node.value) {}

    int dynamicCast::nodeType() const {
      return exprNodeType::dynamicCast;
    }

    void dynamicCast::print(printer_t &pout) const {
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
      value(value_) {}

    parenthesesNode::parenthesesNode(const parenthesesNode &node) :
      value(node.value) {}

    int parenthesesNode::nodeType() const {
      return exprNodeType::parentheses;
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
      blocks(blocks_),
      threads(threads_) {}

    cudaCallNode::cudaCallNode(const cudaCallNode &node) :
      blocks(node.blocks),
      threads(node.threads) {}

    int cudaCallNode::nodeType() const {
      return exprNodeType::cudaCall;
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
