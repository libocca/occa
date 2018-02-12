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
#ifndef OCCA_PARSER_OPERATOR_HEADER2
#define OCCA_PARSER_OPERATOR_HEADER2

#include "occa/types.hpp"
#include "printer.hpp"
#include "occa/parser/primitive.hpp"

namespace occa {
  namespace lang {
    typedef uint64_t opType_t;

    namespace operatorType {
      extern const opType_t not_;
      extern const opType_t positive;
      extern const opType_t negative;
      extern const opType_t tilde;
      extern const opType_t leftIncrement;
      extern const opType_t rightIncrement;
      extern const opType_t increment;
      extern const opType_t leftDecrement;
      extern const opType_t rightDecrement;
      extern const opType_t decrement;

      extern const opType_t add;
      extern const opType_t sub;
      extern const opType_t mult;
      extern const opType_t div;
      extern const opType_t mod;
      extern const opType_t arithmetic;

      extern const opType_t lessThan;
      extern const opType_t lessThanEq;
      extern const opType_t equal;
      extern const opType_t notEqual;
      extern const opType_t greaterThan;
      extern const opType_t greaterThanEq;
      extern const opType_t comparison;

      extern const opType_t and_;
      extern const opType_t or_;
      extern const opType_t boolean;

      extern const opType_t bitAnd;
      extern const opType_t bitOr;
      extern const opType_t xor_;
      extern const opType_t leftShift;
      extern const opType_t rightShift;
      extern const opType_t shift;
      extern const opType_t bitOp;

      extern const opType_t assign;
      extern const opType_t addEq;
      extern const opType_t subEq;
      extern const opType_t multEq;
      extern const opType_t divEq;
      extern const opType_t modEq;
      extern const opType_t andEq;
      extern const opType_t orEq;
      extern const opType_t xorEq;
      extern const opType_t leftShiftEq;
      extern const opType_t rightShiftEq;
      extern const opType_t assignment;

      extern const opType_t comma;
      extern const opType_t scope;
      extern const opType_t dot;
      extern const opType_t dotStar;
      extern const opType_t arrow;
      extern const opType_t arrowStar;

      extern const opType_t leftUnary;

      extern const opType_t rightUnary;

      extern const opType_t binary;

      extern const opType_t ternary;
      extern const opType_t colon;

      extern const opType_t braceStart;
      extern const opType_t braceEnd;
      extern const opType_t bracketStart;
      extern const opType_t bracketEnd;
      extern const opType_t parenthesesStart;
      extern const opType_t parenthesesEnd;

      extern const opType_t braces;
      extern const opType_t brackets;
      extern const opType_t parentheses;

      extern const opType_t pair;
      extern const opType_t pairStart;
      extern const opType_t pairEnd;

      extern const opType_t lineComment;
      extern const opType_t blockCommentStart;
      extern const opType_t blockCommentEnd;
      extern const opType_t comment;

      extern const opType_t hash;
      extern const opType_t hashhash;
      extern const opType_t preprocessor;

      extern const opType_t semicolon;
      extern const opType_t ellipsis;

      extern const opType_t special;

      extern const opType_t overloadable;
    }

    class operator_t {
    public:
      std::string str;
      opType_t opType;
      int precedence;

      operator_t(const std::string &str_,
                 opType_t opType_,
                 int precedence_);

      void print(printer &pout) const;
    };

    class unaryOperator_t : public operator_t {
    public:
      unaryOperator_t(const std::string &str_,
                      opType_t opType_,
                      int precedence_);

      primitive operator () (primitive &value) const;
    };

    class binaryOperator_t : public operator_t {
    public:
      binaryOperator_t(const std::string &str_,
                       opType_t opType_,
                       int precedence_);

      primitive operator () (primitive &leftValue,
                             primitive &rightValue) const;
    };

    namespace op {
      //---[ Left Unary ]---------------
      extern const unaryOperator_t not_;
      extern const unaryOperator_t positive;
      extern const unaryOperator_t negative;
      extern const unaryOperator_t tilde;
      extern const unaryOperator_t leftIncrement;
      extern const unaryOperator_t leftDecrement;
      //================================

      //---[ Right Unary ]--------------
      extern const unaryOperator_t rightIncrement;
      extern const unaryOperator_t rightDecrement;
      //================================

      //---[ Binary ]-------------------
      extern const binaryOperator_t add;
      extern const binaryOperator_t sub;
      extern const binaryOperator_t mult;
      extern const binaryOperator_t div;
      extern const binaryOperator_t mod;

      extern const binaryOperator_t lessThan;
      extern const binaryOperator_t lessThanEq;
      extern const binaryOperator_t equal;
      extern const binaryOperator_t notEqual;
      extern const binaryOperator_t greaterThan;
      extern const binaryOperator_t greaterThanEq;

      extern const binaryOperator_t and_;
      extern const binaryOperator_t or_;
      extern const binaryOperator_t bitAnd;
      extern const binaryOperator_t bitOr;
      extern const binaryOperator_t xor_;
      extern const binaryOperator_t leftShift;
      extern const binaryOperator_t rightShift;

      extern const binaryOperator_t assign;
      extern const binaryOperator_t addEq;
      extern const binaryOperator_t subEq;
      extern const binaryOperator_t multEq;
      extern const binaryOperator_t divEq;
      extern const binaryOperator_t modEq;
      extern const binaryOperator_t andEq;
      extern const binaryOperator_t orEq;
      extern const binaryOperator_t xorEq;
      extern const binaryOperator_t leftShiftEq;
      extern const binaryOperator_t rightShiftEq;

      extern const binaryOperator_t comma;

      // Non-Overloadable
      extern const binaryOperator_t scope;
      extern const binaryOperator_t dot;
      extern const binaryOperator_t dotStar;
      extern const binaryOperator_t arrow;
      extern const binaryOperator_t arrowStar;
      //================================

      //---[ Ternary ]------------------
      extern const operator_t ternary;
      extern const operator_t colon;
      //================================

      //---[ Pairs ]--------------------
      extern const operator_t braceStart;
      extern const operator_t braceEnd;
      extern const operator_t bracketStart;
      extern const operator_t bracketEnd;
      extern const operator_t parenthesesStart;
      extern const operator_t parenthesesEnd;
      //================================

      //---[ Comments ]-----------------
      extern const operator_t lineComment;
      extern const operator_t blockCommentStart;
      extern const operator_t blockCommentEnd;
      //================================

      //---[ Special ]------------------
      extern const operator_t hash;
      extern const operator_t hashhash;

      extern const operator_t semicolon;
      extern const operator_t ellipsis;
      extern const operator_t attribute;
      //================================
    }
  }
}

#endif
