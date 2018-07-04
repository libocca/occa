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
#ifndef OCCA_LANG_EXPRNODE_HEADER
#define OCCA_LANG_EXPRNODE_HEADER

#include <stack>
#include <vector>

#include <occa/lang/operator.hpp>
#include <occa/lang/primitive.hpp>
#include <occa/lang/printer.hpp>
#include <occa/lang/token.hpp>
#include <occa/lang/type.hpp>

namespace occa {
  namespace lang {
    class exprNode;
    class type_t;
    class variable_t;
    class function_t;

    typedef std::vector<exprNode*>  exprNodeVector;
    typedef std::vector<exprNode**> exprNodeRefVector;
    typedef std::stack<exprNode*>   exprNodeStack;
    typedef std::vector<token_t*>   tokenVector;

    namespace exprNodeType {
      extern const udim_t empty;
      extern const udim_t primitive;
      extern const udim_t char_;
      extern const udim_t string;
      extern const udim_t identifier;
      extern const udim_t type;
      extern const udim_t vartype;
      extern const udim_t variable;
      extern const udim_t function;
      extern const udim_t value;

      extern const udim_t leftUnary;
      extern const udim_t rightUnary;
      extern const udim_t binary;
      extern const udim_t ternary;
      extern const udim_t op;

      extern const udim_t pair;

      extern const udim_t subscript;
      extern const udim_t call;

      extern const udim_t sizeof_;
      extern const udim_t sizeof_pack_;
      extern const udim_t new_;
      extern const udim_t delete_;
      extern const udim_t throw_;

      extern const udim_t typeid_;
      extern const udim_t noexcept_;
      extern const udim_t alignof_;

      extern const udim_t const_cast_;
      extern const udim_t dynamic_cast_;
      extern const udim_t static_cast_;
      extern const udim_t reinterpret_cast_;

      extern const udim_t funcCast;
      extern const udim_t parenCast;
      extern const udim_t constCast;
      extern const udim_t staticCast;
      extern const udim_t reinterpretCast;
      extern const udim_t dynamicCast;

      extern const udim_t parentheses;
      extern const udim_t tuple;
      extern const udim_t cudaCall;
    }

    class exprNode {
    public:
      token_t *token;

      exprNode(token_t *token_);

      virtual ~exprNode();

      template <class TM>
      inline bool is() const {
        return (dynamic_cast<const TM*>(this) != NULL);
      }

      template <class TM>
      inline TM& to() {
        TM *ptr = dynamic_cast<TM*>(this);
        OCCA_ERROR("Unable to cast exprNode::to",
                   ptr != NULL);
        return *ptr;
      }

      template <class TM>
      inline const TM& to() const {
        const TM *ptr = dynamic_cast<const TM*>(this);
        OCCA_ERROR("Unable to cast exprNode::to",
                   ptr != NULL);
        return *ptr;
      }

      virtual udim_t type() const = 0;

      virtual exprNode* clone() const = 0;
      static exprNode* clone(exprNode *expr);

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children) = 0;

      virtual bool hasAttribute(const std::string &attr) const;

      virtual variable_t* getVariable();

      virtual exprNode* wrapInParentheses();

      virtual void print(printer &pout) const = 0;

      std::string toString() const;

      void printWarning(const std::string &message) const;
      void printError(const std::string &message) const;

      void debugPrint() const;

      virtual void debugPrint(const std::string &prefix) const = 0;

      void childDebugPrint(const std::string &prefix) const;
    };

    printer& operator << (printer &pout,
                          const exprNode &node);

    void cloneExprNodeVector(exprNodeVector &dest,
                             const exprNodeVector &src);

    void freeExprNodeVector(exprNodeVector &vec);

    //---[ Empty ]----------------------
    class emptyNode : public exprNode {
    public:
      emptyNode();
      virtual ~emptyNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    extern const emptyNode noExprNode;
    //==================================

    //---[ Values ]---------------------
    class primitiveNode : public exprNode {
    public:
      primitive value;

      primitiveNode(token_t *token_,
                    primitive value_);

      primitiveNode(const primitiveNode& node);

      virtual ~primitiveNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class charNode : public exprNode {
    public:
      std::string value;

      charNode(token_t *token_,
               const std::string &value_);

      charNode(const charNode& node);

      virtual ~charNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class stringNode : public exprNode {
    public:
      int encoding;
      std::string value;

      stringNode(token_t *token_,
                 const std::string &value_);

      stringNode(const stringNode& node);

      virtual ~stringNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class identifierNode : public exprNode {
    public:
      std::string value;

      identifierNode(token_t *token_,
                     const std::string &value_);

      identifierNode(const identifierNode& node);

      virtual ~identifierNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    //  |---[ Type ]--------------------
    class typeNode : public exprNode {
    public:
      type_t &value;

      typeNode(token_t *token_,
               type_t &value_);

      typeNode(const typeNode& node);

      virtual ~typeNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual bool hasAttribute(const std::string &attr) const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
    //  |===============================

    //  |---[ Vartype ]-----------------
    class vartypeNode : public exprNode {
    public:
      vartype_t value;

      vartypeNode(token_t *token_,
                  const vartype_t &value_);

      vartypeNode(const vartypeNode& node);

      virtual ~vartypeNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual bool hasAttribute(const std::string &attr) const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
    //  |===============================

    //  |---[ Variable ]----------------
    class variableNode : public exprNode {
    public:
      variable_t &value;

      variableNode(token_t *token_,
                   variable_t &value_);

      variableNode(const variableNode& node);

      virtual ~variableNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual bool hasAttribute(const std::string &attr) const;

      virtual variable_t* getVariable();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
    //  |===============================

    //  |---[ Function ]----------------
    class functionNode : public exprNode {
    public:
      function_t &value;

      functionNode(token_t *token_,
                   function_t &value_);

      functionNode(const functionNode& node);

      virtual ~functionNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual bool hasAttribute(const std::string &attr) const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
    //  |===============================
    //==================================

    //---[ Operators ]------------------
    class exprOpNode : public exprNode {
    public:
      const operator_t &op;

      exprOpNode(operatorToken &token_);

      exprOpNode(token_t *token_,
                 const operator_t &op_);

      opType_t opType() const;

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual exprNode* wrapInParentheses();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class leftUnaryOpNode : public exprOpNode {
    public:
      exprNode *value;

      leftUnaryOpNode(token_t *token_,
                      const unaryOperator_t &op_,
                      const exprNode &value_);

      leftUnaryOpNode(const leftUnaryOpNode &node);

      virtual ~leftUnaryOpNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual variable_t* getVariable();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class rightUnaryOpNode : public exprOpNode {
    public:
      exprNode *value;

      rightUnaryOpNode(const unaryOperator_t &op_,
                       const exprNode &value_);

      rightUnaryOpNode(token_t *token,
                       const unaryOperator_t &op_,
                       const exprNode &value_);

      rightUnaryOpNode(const rightUnaryOpNode &node);

      virtual ~rightUnaryOpNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual exprNode* startNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual variable_t* getVariable();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class binaryOpNode : public exprOpNode {
    public:
      exprNode *leftValue, *rightValue;

      binaryOpNode(const binaryOperator_t &op_,
                   const exprNode &leftValue_,
                   const exprNode &rightValue_);

      binaryOpNode(token_t *token,
                   const binaryOperator_t &op_,
                   const exprNode &leftValue_,
                   const exprNode &rightValue_);

      binaryOpNode(const binaryOpNode &node);

      virtual ~binaryOpNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual variable_t* getVariable();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class ternaryOpNode : public exprOpNode {
    public:
      exprNode *checkValue, *trueValue, *falseValue;

      ternaryOpNode(const exprNode &checkValue_,
                    const exprNode &trueValue_,
                    const exprNode &falseValue_);

      ternaryOpNode(const ternaryOpNode &node);
      virtual ~ternaryOpNode();

      virtual udim_t type() const;
      opType_t opType() const;

      virtual exprNode* clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
    //==================================

    //---[ Pseudo Operators ]-----------
    class subscriptNode : public exprNode {
    public:
      exprNode *value, *index;

      subscriptNode(token_t *token_,
                    const exprNode &value_,
                    const exprNode &index_);

      subscriptNode(const subscriptNode &node);

      virtual ~subscriptNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual variable_t* getVariable();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class callNode : public exprNode {
    public:
      exprNode *value;
      exprNodeVector args;

      callNode(token_t *token_,
               const exprNode &value_,
               const exprNodeVector &args_);

      callNode(const callNode &node);

      virtual ~callNode();

      inline int argCount() const {
        return (int) args.size();
      }

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class newNode : public exprNode {
    public:
      vartype_t valueType;
      exprNode *value;
      exprNode *size;

      newNode(token_t *token_,
              const vartype_t &valueType_,
              const exprNode &value_);

      newNode(token_t *token_,
              const vartype_t &valueType_,
              const exprNode &value_,
              const exprNode &size_);

      newNode(const newNode &node);

      virtual ~newNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual exprNode* wrapInParentheses();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class deleteNode : public exprNode {
    public:
      exprNode *value;
      bool isArray;

      deleteNode(token_t *token_,
                 const exprNode &value_,
                 const bool isArray_);

      deleteNode(const deleteNode &node);

      virtual ~deleteNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual exprNode* wrapInParentheses();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class throwNode : public exprNode {
    public:
      exprNode *value;

      throwNode(token_t *token_,
                const exprNode &value_);

      throwNode(const throwNode &node);

      virtual ~throwNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual exprNode* wrapInParentheses();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
    //==================================

    //---[ Builtins ]-------------------
    class sizeofNode : public exprNode {
    public:
      exprNode *value;

      sizeofNode(token_t *token_,
                 const exprNode &value_);

      sizeofNode(const sizeofNode &node);

      virtual ~sizeofNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* endNode();

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class funcCastNode : public exprNode {
    public:
      vartype_t valueType;
      exprNode *value;

      funcCastNode(token_t *token_,
                   const vartype_t &valueType_,
                   const exprNode &value_);

      funcCastNode(const funcCastNode &node);

      virtual ~funcCastNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class parenCastNode : public exprNode {
    public:
      vartype_t valueType;
      exprNode *value;

      parenCastNode(token_t *token_,
                    const vartype_t &valueType_,
                    const exprNode &value_);

      parenCastNode(const parenCastNode &node);

      virtual ~parenCastNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual exprNode* wrapInParentheses();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class constCastNode : public exprNode {
    public:
      vartype_t valueType;
      exprNode *value;

      constCastNode(token_t *token_,
                    const vartype_t &valueType_,
                    const exprNode &value_);

      constCastNode(const constCastNode &node);

      virtual ~constCastNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class staticCastNode : public exprNode {
    public:
      vartype_t valueType;
      exprNode *value;

      staticCastNode(token_t *token_,
                     const vartype_t &valueType_,
                     const exprNode &value_);

      staticCastNode(const staticCastNode &node);

      virtual ~staticCastNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class reinterpretCastNode : public exprNode {
    public:
      vartype_t valueType;
      exprNode *value;

      reinterpretCastNode(token_t *token_,
                          const vartype_t &valueType_,
                          const exprNode &value_);

      reinterpretCastNode(const reinterpretCastNode &node);

      virtual ~reinterpretCastNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class dynamicCastNode : public exprNode {
    public:
      vartype_t valueType;
      exprNode *value;

      dynamicCastNode(token_t *token_,
                      const vartype_t &valueType_,
                      const exprNode &value_);

      dynamicCastNode(const dynamicCastNode &node);

      virtual ~dynamicCastNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
    //==================================

    //---[ Misc ]-----------------------
    class pairNode : public exprNode {
    public:
      const operator_t &op;
      exprNode *value;

      pairNode(operatorToken &opToken,
               const exprNode &value_);

      pairNode(const pairNode &node);

      virtual ~pairNode();

      virtual udim_t type() const;
      opType_t opType() const;

      virtual exprNode* clone() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class parenthesesNode : public exprNode {
    public:
      exprNode *value;

      parenthesesNode(token_t *token_,
                      const exprNode &value_);

      parenthesesNode(const parenthesesNode &node);

      virtual ~parenthesesNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual variable_t* getVariable();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    class tupleNode : public exprNode {
    public:
      exprNodeVector args;

      tupleNode(token_t *token_,
                const exprNodeVector &args_);

      tupleNode(const tupleNode &node);

      virtual ~tupleNode();

      inline int argCount() const {
        return (int) args.size();
      }

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
    //==================================

    //---[ Extensions ]-----------------
    class cudaCallNode : public exprNode {
    public:
      exprNode *value;
      exprNode *blocks, *threads;

      cudaCallNode(token_t *token_,
                   const exprNode &value_,
                   const exprNode &blocks_,
                   const exprNode &threads_);

      cudaCallNode(const cudaCallNode &node);

      virtual ~cudaCallNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
    //==================================
  }
}

#endif
