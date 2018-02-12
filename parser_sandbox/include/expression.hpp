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
#ifndef OCCA_PARSER_EXPRESSION_HEADER2
#define OCCA_PARSER_EXPRESSION_HEADER2

#include <stack>
#include <vector>

#include "occa/parser/primitive.hpp"
#include "token.hpp"
#include "operator.hpp"
#include "variable.hpp"
#include "printer.hpp"

namespace occa {
  namespace lang {
    class exprNode;

    typedef std::vector<exprNode*>     exprNodeVector;
    typedef std::stack<exprNode*>      exprNodeStack;
    typedef std::stack<operatorToken*> operatorStack;
    typedef std::vector<token_t*>      tokenVector;

    class exprNodeType {
    public:
      static const int empty           = (1 << 0);
      static const int primitive       = (1 << 1);
      static const int char_           = (1 << 2);
      static const int string          = (1 << 3);
      static const int identifier      = (1 << 4);
      static const int variable        = (1 << 5);
      static const int value           = (primitive |
                                          variable);
      static const int leftUnary       = (1 << 6);
      static const int rightUnary      = (1 << 7);
      static const int binary          = (1 << 8);
      static const int ternary         = (1 << 9);
      static const int op              = (leftUnary  |
                                          rightUnary |
                                          binary     |
                                          ternary);
      static const int subscript       = (1 << 10);
      static const int call            = (1 << 11);
      static const int new_            = (1 << 12);
      static const int delete_         = (1 << 13);
      static const int throw_          = (1 << 14);
      static const int sizeof_         = (1 << 15);
      static const int funcCast        = (1 << 16);
      static const int parenCast       = (1 << 17);
      static const int constCast       = (1 << 18);
      static const int staticCast      = (1 << 19);
      static const int reinterpretCast = (1 << 20);
      static const int dynamicCast     = (1 << 21);
      static const int parentheses     = (1 << 22);
      static const int tuple           = (1 << 23);
      static const int cudaCall        = (1 << 24);
    };

    class exprNode {
    public:
      token_t *token;

      exprNode(token_t *token_ = NULL);

      virtual ~exprNode();

      virtual int nodeType() const = 0;

      virtual exprNode& clone() const = 0;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual void print(printer &pout) const = 0;

      std::string toString() const;

      void debugPrint() const;

      virtual void debugPrint(const std::string &prefix) const = 0;

      void childDebugPrint(const std::string &prefix) const;

      // Load tokens as an expression
      static exprNode* load(const tokenVector &tokens);

      static void pushOutputNode(token_t *token,
                                 exprNodeStack &output);

      static bool closePair(operatorToken &opToken,
                            exprNodeStack &output,
                            operatorStack &operators);

      static bool applyFasterOperators(operatorToken &opToken,
                                       exprNodeStack &output,
                                       operatorStack &operators);

      static bool applyOperator(operatorToken &opToken,
                                exprNodeStack &output,
                                operatorStack &operators);
    };

    //---[ Empty ]----------------------
    class emptyNode : public exprNode {
    public:
      emptyNode();
      ~emptyNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
    //==================================

    //---[ Values ]---------------------
    class primitiveNode : public exprNode {
    public:
      primitive value;

      primitiveNode(primitive value_);

      primitiveNode(token_t *token_,
                    primitive value_);

      primitiveNode(const primitiveNode& node);

      ~primitiveNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class charNode : public exprNode {
    public:
      std::string value;

      charNode(const std::string &value_);

      charNode(token_t *token_,
               const std::string &value_);

      charNode(const charNode& node);

      ~charNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class stringNode : public exprNode {
    public:
      int encoding;
      std::string value;

      stringNode(const std::string &value_);

      stringNode(token_t *token_,
                 const std::string &value_);

      stringNode(const stringNode& node);

      ~stringNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class identifierNode : public exprNode {
    public:
      std::string value;

      identifierNode(const std::string &value_);

      identifierNode(token_t *token_,
                     const std::string &value_);

      identifierNode(const identifierNode& node);

      ~identifierNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class variableNode : public exprNode {
    public:
      variable &value;

      variableNode(variable &value_);

      variableNode(token_t *token_,
                   variable &value_);

      variableNode(const variableNode& node);

      ~variableNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
    //==================================

    //---[ Operators ]------------------
    class leftUnaryOpNode : public exprNode {
    public:
      const unaryOperator_t &op;
      exprNode &value;

      leftUnaryOpNode(const unaryOperator_t &op_,
                      exprNode &value_);

      leftUnaryOpNode(token_t *token_,
                      const unaryOperator_t &op_,
                      exprNode &value_);

      leftUnaryOpNode(const leftUnaryOpNode &node);

      ~leftUnaryOpNode();

      virtual int nodeType() const;
      opType_t opnodeType() const;

      virtual exprNode& clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class rightUnaryOpNode : public exprNode {
    public:
      const unaryOperator_t &op;
      exprNode &value;

      rightUnaryOpNode(const unaryOperator_t &op_,
                       exprNode &value_);

      rightUnaryOpNode(token_t *token,
                       const unaryOperator_t &op_,
                       exprNode &value_);

      rightUnaryOpNode(const rightUnaryOpNode &node);

      ~rightUnaryOpNode();

      virtual int nodeType() const;
      opType_t opnodeType() const;

      virtual exprNode& clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class binaryOpNode : public exprNode {
    public:
      const binaryOperator_t &op;
      exprNode &leftValue, &rightValue;

      binaryOpNode(const binaryOperator_t &op_,
                   exprNode &leftValue_,
                   exprNode &rightValue_);

      binaryOpNode(token_t *token,
                   const binaryOperator_t &op_,
                   exprNode &leftValue_,
                   exprNode &rightValue_);

      binaryOpNode(const binaryOpNode &node);

      ~binaryOpNode();

      virtual int nodeType() const;
      opType_t opnodeType() const;

      virtual exprNode& clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class ternaryOpNode : public exprNode {
    public:
      exprNode &checkValue, &trueValue, &falseValue;

      ternaryOpNode(exprNode &checkValue_,
                    exprNode &trueValue_,
                    exprNode &falseValue_);

      ternaryOpNode(token_t *token,
                    exprNode &checkValue_,
                    exprNode &trueValue_,
                    exprNode &falseValue_);

      ternaryOpNode(const ternaryOpNode &node);
      ~ternaryOpNode();

      virtual int nodeType() const;
      opType_t opnodeType() const;

      virtual exprNode& clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
    //==================================

    //---[ Pseudo Operators ]-----------
    class subscriptNode : public exprNode {
    public:
      exprNode &value, &index;

      subscriptNode(exprNode &value_,
                    exprNode &index_);

      subscriptNode(token_t *token_,
                    exprNode &value_,
                    exprNode &index_);

      subscriptNode(const subscriptNode &node);

      ~subscriptNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class callNode : public exprNode {
    public:
      exprNode &value;
      exprNodeVector args;

      callNode(exprNode &value_,
               exprNodeVector args_);

      callNode(token_t *token_,
               exprNode &value_,
               exprNodeVector args_);

      callNode(const callNode &node);

      ~callNode();

      inline int argCount() const {
        return (int) args.size();
      }

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
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

      newNode(token_t *token_,
              type_t &type_,
              exprNode &value_,
              exprNode &size_);

      newNode(const newNode &node);

      ~newNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class deleteNode : public exprNode {
    public:
      exprNode &value;
      bool isArray;

      deleteNode(exprNode &value_,
                 const bool isArray_);

      deleteNode(token_t *token_,
                 exprNode &value_,
                 const bool isArray_);

      deleteNode(const deleteNode &node);

      ~deleteNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class throwNode : public exprNode {
    public:
      exprNode &value;

      throwNode(exprNode &value_);

      throwNode(token_t *token_,
                exprNode &value_);

      throwNode(const throwNode &node);

      ~throwNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
    //==================================

    //---[ Builtins ]-------------------
    class sizeofNode : public exprNode {
    public:
      exprNode &value;

      sizeofNode(exprNode &value_);

      sizeofNode(token_t *token_,
                 exprNode &value_);

      sizeofNode(const sizeofNode &node);

      ~sizeofNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class funcCastNode : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      funcCastNode(type_t &type_,
                   exprNode &value_);

      funcCastNode(token_t *token_,
                   type_t &type_,
                   exprNode &value_);

      funcCastNode(const funcCastNode &node);

      ~funcCastNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class parenCastNode : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      parenCastNode(type_t &type_,
                    exprNode &value_);

      parenCastNode(token_t *token_,
                    type_t &type_,
                    exprNode &value_);

      parenCastNode(const parenCastNode &node);

      ~parenCastNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class constCastNode : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      constCastNode(type_t &type_,
                    exprNode &value_);

      constCastNode(token_t *token_,
                    type_t &type_,
                    exprNode &value_);

      constCastNode(const constCastNode &node);

      ~constCastNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class staticCastNode : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      staticCastNode(type_t &type_,
                     exprNode &value_);

      staticCastNode(token_t *token_,
                     type_t &type_,
                     exprNode &value_);

      staticCastNode(const staticCastNode &node);

      ~staticCastNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class reinterpretCastNode : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      reinterpretCastNode(type_t &type_,
                          exprNode &value_);

      reinterpretCastNode(token_t *token_,
                          type_t &type_,
                          exprNode &value_);

      reinterpretCastNode(const reinterpretCastNode &node);

      ~reinterpretCastNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class dynamicCastNode : public exprNode {
    public:
      type_t &type;
      exprNode &value;

      dynamicCastNode(type_t &type_,
                      exprNode &value_);

      dynamicCastNode(token_t *token_,
                      type_t &type_,
                      exprNode &value_);

      dynamicCastNode(const dynamicCastNode &node);

      ~dynamicCastNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
    //==================================

    //---[ Misc ]-----------------------
    class parenthesesNode : public exprNode {
    public:
      exprNode &value;

      parenthesesNode(exprNode &value_);

      parenthesesNode(token_t *token_,
                      exprNode &value_);

      parenthesesNode(const parenthesesNode &node);

      ~parenthesesNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
    //==================================

    //---[ Extensions ]-----------------
    class cudaCallNode : public exprNode {
    public:
      exprNode &blocks, &threads;

      cudaCallNode(exprNode &blocks_,
                   exprNode &threads_);

      cudaCallNode(token_t *token_,
                   exprNode &blocks_,
                   exprNode &threads_);

      cudaCallNode(const cudaCallNode &node);

      ~cudaCallNode();

      virtual int nodeType() const;

      virtual exprNode& clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
    //==================================
  }
}

#endif
