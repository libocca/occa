#ifndef OCCA_PARSER_EXPRESSION_HEADER2
#define OCCA_PARSER_EXPRESSION_HEADER2

#include <vector>

#include "occa/tools/gc.hpp"
#include "occa/parser/primitive.hpp"
#include "operator.hpp"
#include "variable.hpp"

namespace occa {
  namespace lang {
    class exprNode;

    typedef std::vector<exprNode*> exprNodeVector_t;

    class exprNodeType {
    public:
      static const int empty           = (1 << 0);
      static const int primitive       = (1 << 1);
      static const int variable        = (1 << 2);
      static const int function        = (1 << 3);
      static const int value           = (primitive |
                                          variable  |
                                          function);
      static const int leftUnary       = (1 << 4);
      static const int rightUnary      = (1 << 5);
      static const int binary          = (1 << 6);
      static const int ternary         = (1 << 7);
      static const int op              = (leftUnary  |
                                          rightUnary |
                                          binary     |
                                          ternary);
      static const int subscript       = (1 << 8);
      static const int call            = (1 << 9);
      static const int new_            = (1 << 10);
      static const int delete_         = (1 << 11);
      static const int throw_          = (1 << 12);
      static const int sizeof_         = (1 << 13);
      static const int funcCast        = (1 << 14);
      static const int parenCast       = (1 << 15);
      static const int constCast       = (1 << 16);
      static const int staticCast      = (1 << 17);
      static const int reinterpretCast = (1 << 18);
      static const int dynamicCast     = (1 << 19);
      static const int parentheses     = (1 << 20);
      static const int tuple           = (1 << 21);
      static const int cudaCall        = (1 << 22);
    };

    class exprNode {
    public:
      virtual int nodeType() const = 0;
      virtual void print(printer_t &pout) const = 0;
    };

    //---[ Empty ]----------------------
    class emptyNode : public exprNode {
    public:
      emptyNode();

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };
    //==================================

    //---[ Values ]---------------------
    class primitiveNode : public exprNode {
    public:
      primitive value;

      primitiveNode(primitive value_);
      primitiveNode(const primitiveNode& node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };

    class variableNode : public exprNode {
    public:
      variable_t &value;

      variableNode(variable_t &value_);
      variableNode(const variableNode& node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };

    class functionNode : public exprNode {
    public:
      variable_t &value;

      functionNode(variable_t &value_);
      functionNode(const functionNode &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };
    //==================================

    //---[ Operators ]------------------
    class opNode : virtual public exprNode {
    public:
      operator_t &op;

      opNode(operator_t &op_);
      int opnodeType() const;
    };

    class leftUnaryOpNode : public opNode {
    public:
      exprNode &value;

      leftUnaryOpNode(operator_t &op_,
                      exprNode &value_);
      leftUnaryOpNode(const leftUnaryOpNode &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };

    class rightUnaryOpNode : public opNode {
    public:
      exprNode &value;

      rightUnaryOpNode(operator_t &op_,
                       exprNode &value_);
      rightUnaryOpNode(const rightUnaryOpNode &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };

    class binaryOpNode : public opNode {
    public:
      exprNode &leftValue, &rightValue;

      binaryOpNode(operator_t &op_,
                   exprNode &leftValue_,
                   exprNode &rightValue_);
      binaryOpNode(const binaryOpNode &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };

    class ternaryOpNode : public opNode {
    public:
      exprNode &checkValue, &trueValue, &falseValue;

      ternaryOpNode(operator_t &op_,
                    exprNode &checkValue_,
                    exprNode &trueValue_,
                    exprNode &falseValue_);
      ternaryOpNode(const ternaryOpNode &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };
    //==================================

    //---[ Pseudo Operators ]-----------
    class subscriptNode : public exprNode {
    public:
      exprNode &value, &index;

      subscriptNode(exprNode &value_,
                    exprNode &index_);
      subscriptNode(const subscriptNode &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };

    class callNode : public exprNode {
    public:
      exprNode &value;
      exprNodeVector_t args;

      callNode(exprNode &value_,
               exprNodeVector_t args_);
      callNode(const callNode &node);

      inline int argCount() const {
        return (int) args.size();
      }

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };

    class newNode : public exprNode {
    public:
      type_t &type;
      exprNode &size;

      newNode(type_t &type_,
              exprNode &size_);
      newNode(const newNode &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };

    class deleteNode : public exprNode {
    public:
      exprNode &value;
      bool isArray;

      deleteNode(exprNode &value_,
                 const bool isArray_);
      deleteNode(const deleteNode &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };

    class throwNode : public exprNode {
    public:
      exprNode &value;

      throwNode(exprNode &value_);
      throwNode(const throwNode &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };
    //==================================

    //---[ Builtins ]-------------------
    class sizeofNode : public exprNode {
    public:
      exprNode &value;

      sizeofNode(exprNode &value_);
      sizeofNode(const sizeofNode &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };

    class funcCastNode : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      funcCastNode(type_t &type_,
                   exprNode &value_);
      funcCastNode(const funcCastNode &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };

    class parenCastNode : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      parenCastNode(type_t &type_,
                    exprNode &value_);
      parenCastNode(const parenCastNode &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };

    class constCast : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      constCast(type_t &type_,
                exprNode &value_);
      constCast(const constCast &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };

    class staticCast : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      staticCast(type_t &type_,
                 exprNode &value_);
      staticCast(const staticCast &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };

    class reinterpretCast : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      reinterpretCast(type_t &type_,
                      exprNode &value_);
      reinterpretCast(const reinterpretCast &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };

    class dynamicCast : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      dynamicCast(type_t &type_,
                  exprNode &value_);
      dynamicCast(const dynamicCast &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };
    //==================================

    //---[ Misc ]-----------------------
    class parenthesesNode : public exprNode {
    public:
      exprNode &value;

      parenthesesNode(exprNode &value_);
      parenthesesNode(const parenthesesNode &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };
    //==================================

    //---[ Extensions ]-----------------
    class cudaCallNode : public exprNode {
    public:
      exprNode &blocks, &threads;

      cudaCallNode(exprNode &blocks_,
                   exprNode &threads_);
      cudaCallNode(const cudaCallNode &node);

      virtual int nodeType() const;
      virtual void print(printer_t &pout) const;
    };
    //==================================
  }
}

#endif
