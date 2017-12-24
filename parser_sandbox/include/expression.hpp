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
      static const int value           = (primitive |
                                          variable);
      static const int leftUnary       = (1 << 5);
      static const int rightUnary      = (1 << 6);
      static const int binary          = (1 << 7);
      static const int ternary         = (1 << 8);
      static const int op              = (leftUnary  |
                                          rightUnary |
                                          binary     |
                                          ternary);
      static const int subscript       = (1 << 9);
      static const int call            = (1 << 10);
      static const int new_            = (1 << 11);
      static const int delete_         = (1 << 12);
      static const int throw_          = (1 << 13);
      static const int sizeof_         = (1 << 14);
      static const int funcCast        = (1 << 15);
      static const int parenCast       = (1 << 16);
      static const int constCast       = (1 << 17);
      static const int staticCast      = (1 << 18);
      static const int reinterpretCast = (1 << 19);
      static const int dynamicCast     = (1 << 20);
      static const int parentheses     = (1 << 21);
      static const int tuple           = (1 << 22);
      static const int cudaCall        = (1 << 23);
    };

    class exprNode {
    public:
      virtual ~exprNode();

      virtual int nodeType() const = 0;
      virtual exprNode& clone() const = 0;
      virtual void print(printer &pout) const = 0;

      std::string toString() const;
      void debugPrint() const;
    };

    //---[ Empty ]----------------------
    class emptyNode : public exprNode {
    public:
      emptyNode();
      ~emptyNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ Values ]---------------------
    class primitiveNode : public exprNode {
    public:
      primitive value;

      primitiveNode(primitive value_);
      primitiveNode(const primitiveNode& node);
      ~primitiveNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };

    class variableNode : public exprNode {
    public:
      variable &value;

      variableNode(variable &value_);
      variableNode(const variableNode& node);
      ~variableNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ Operators ]------------------
    class opNode : virtual public exprNode {
    public:
      const operator_t &op;

      opNode(const operator_t &op_);
      int opnodeType() const;
    };

    class leftUnaryOpNode : public opNode {
    public:
      exprNode &value;

      leftUnaryOpNode(const operator_t &op_,
                      exprNode &value_);
      leftUnaryOpNode(const leftUnaryOpNode &node);
      ~leftUnaryOpNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };

    class rightUnaryOpNode : public opNode {
    public:
      exprNode &value;

      rightUnaryOpNode(const operator_t &op_,
                       exprNode &value_);
      rightUnaryOpNode(const rightUnaryOpNode &node);
      ~rightUnaryOpNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };

    class binaryOpNode : public opNode {
    public:
      exprNode &leftValue, &rightValue;

      binaryOpNode(const operator_t &op_,
                   exprNode &leftValue_,
                   exprNode &rightValue_);
      binaryOpNode(const binaryOpNode &node);
      ~binaryOpNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };

    class ternaryOpNode : public opNode {
    public:
      exprNode &checkValue, &trueValue, &falseValue;

      ternaryOpNode(const operator_t &op_,
                    exprNode &checkValue_,
                    exprNode &trueValue_,
                    exprNode &falseValue_);
      ternaryOpNode(const ternaryOpNode &node);
      ~ternaryOpNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ Pseudo Operators ]-----------
    class subscriptNode : public exprNode {
    public:
      exprNode &value, &index;

      subscriptNode(exprNode &value_,
                    exprNode &index_);
      subscriptNode(const subscriptNode &node);
      ~subscriptNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };

    class callNode : public exprNode {
    public:
      exprNode &value;
      exprNodeVector_t args;

      callNode(exprNode &value_,
               exprNodeVector_t args_);
      callNode(const callNode &node);
      ~callNode();

      inline int argCount() const {
        return (int) args.size();
      }

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };

    class newNode : public exprNode {
    public:
      type_t &type;
      exprNode &value;
      exprNode &size;

      newNode(type_t &type_,
              exprNode &value_);
      newNode(type_t &type_,
              exprNode &value_,
              exprNode &size_);
      newNode(const newNode &node);
      ~newNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };

    class deleteNode : public exprNode {
    public:
      exprNode &value;
      bool isArray;

      deleteNode(exprNode &value_,
                 const bool isArray_);
      deleteNode(const deleteNode &node);
      ~deleteNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };

    class throwNode : public exprNode {
    public:
      exprNode &value;

      throwNode(exprNode &value_);
      throwNode(const throwNode &node);
      ~throwNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ Builtins ]-------------------
    class sizeofNode : public exprNode {
    public:
      exprNode &value;

      sizeofNode(exprNode &value_);
      sizeofNode(const sizeofNode &node);
      ~sizeofNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };

    class funcCastNode : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      funcCastNode(type_t &type_,
                   exprNode &value_);
      funcCastNode(const funcCastNode &node);
      ~funcCastNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };

    class parenCastNode : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      parenCastNode(type_t &type_,
                    exprNode &value_);
      parenCastNode(const parenCastNode &node);
      ~parenCastNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };

    class constCastNode : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      constCastNode(type_t &type_,
                    exprNode &value_);
      constCastNode(const constCastNode &node);
      ~constCastNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };

    class staticCastNode : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      staticCastNode(type_t &type_,
                     exprNode &value_);
      staticCastNode(const staticCastNode &node);
      ~staticCastNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };

    class reinterpretCastNode : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      reinterpretCastNode(type_t &type_,
                          exprNode &value_);
      reinterpretCastNode(const reinterpretCastNode &node);
      ~reinterpretCastNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };

    class dynamicCastNode : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      dynamicCastNode(type_t &type_,
                      exprNode &value_);
      dynamicCastNode(const dynamicCastNode &node);
      ~dynamicCastNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ Misc ]-----------------------
    class parenthesesNode : public exprNode {
    public:
      exprNode &value;

      parenthesesNode(exprNode &value_);
      parenthesesNode(const parenthesesNode &node);
      ~parenthesesNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ Extensions ]-----------------
    class cudaCallNode : public exprNode {
    public:
      exprNode &blocks, &threads;

      cudaCallNode(exprNode &blocks_,
                   exprNode &threads_);
      cudaCallNode(const cudaCallNode &node);
      ~cudaCallNode();

      virtual int nodeType() const;
      virtual exprNode& clone() const;
      virtual void print(printer &pout) const;
    };
    //==================================
  }
}

#endif
