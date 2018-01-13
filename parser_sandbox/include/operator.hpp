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

namespace occa {
  namespace lang {
    typedef uint64_t optype_t;

    namespace operatorType {
      extern const optype_t not_;
      extern const optype_t positive;
      extern const optype_t negative;
      extern const optype_t tilde;
      extern const optype_t leftIncrement;
      extern const optype_t rightIncrement;
      extern const optype_t increment;
      extern const optype_t leftDecrement;
      extern const optype_t rightDecrement;
      extern const optype_t decrement;

      extern const optype_t add;
      extern const optype_t sub;
      extern const optype_t mult;
      extern const optype_t div;
      extern const optype_t mod;
      extern const optype_t arithmetic;

      extern const optype_t lessThan;
      extern const optype_t lessThanEq;
      extern const optype_t equal;
      extern const optype_t notEqual;
      extern const optype_t greaterThan;
      extern const optype_t greaterThanEq;
      extern const optype_t comparison;

      extern const optype_t and_;
      extern const optype_t or_;
      extern const optype_t boolean;

      extern const optype_t bitAnd;
      extern const optype_t bitOr;
      extern const optype_t xor_;
      extern const optype_t leftShift;
      extern const optype_t rightShift;
      extern const optype_t shift;
      extern const optype_t bitOp;

      extern const optype_t assign;
      extern const optype_t addEq;
      extern const optype_t subEq;
      extern const optype_t multEq;
      extern const optype_t divEq;
      extern const optype_t modEq;
      extern const optype_t andEq;
      extern const optype_t orEq;
      extern const optype_t xorEq;
      extern const optype_t leftShiftEq;
      extern const optype_t rightShiftEq;
      extern const optype_t assignment;

      extern const optype_t comma;
      extern const optype_t scope;
      extern const optype_t dot;
      extern const optype_t dotStar;
      extern const optype_t arrow;
      extern const optype_t arrowStar;

      extern const optype_t leftUnary;

      extern const optype_t rightUnary;

      extern const optype_t binary;

      extern const optype_t ternary;
      extern const optype_t colon;

      extern const optype_t braceStart;
      extern const optype_t braceEnd;
      extern const optype_t bracketStart;
      extern const optype_t bracketEnd;
      extern const optype_t parenthesesStart;
      extern const optype_t parenthesesEnd;

      extern const optype_t braces;
      extern const optype_t brackets;
      extern const optype_t parentheses;

      extern const optype_t pair;
      extern const optype_t pairStart;
      extern const optype_t pairEnd;

      extern const optype_t hash;
      extern const optype_t hashhash;
      extern const optype_t preprocessor;

      extern const optype_t semicolon;
      extern const optype_t ellipsis;

      extern const optype_t special;

      extern const optype_t overloadable;
    }

    class operator_t {
    public:
      std::string str;
      optype_t optype;
      int precedence;

      operator_t(const std::string &str_,
                 optype_t optype_,
                 int precedence_);

      void print(printer &pout) const;
    };

    namespace op {
      //---[ Left Unary ]---------------
      extern const operator_t not_;
      extern const operator_t positive;
      extern const operator_t negative;
      extern const operator_t tilde;
      extern const operator_t leftIncrement;
      extern const operator_t leftDecrement;
      //================================

      //---[ Right Unary ]--------------
      extern const operator_t rightIncrement;
      extern const operator_t rightDecrement;
      //================================

      //---[ Binary ]-------------------
      extern const operator_t add;
      extern const operator_t sub;
      extern const operator_t mult;
      extern const operator_t div;
      extern const operator_t mod;

      extern const operator_t lessThan;
      extern const operator_t lessThanEq;
      extern const operator_t equal;
      extern const operator_t notEqual;
      extern const operator_t greaterThan;
      extern const operator_t greaterThanEq;

      extern const operator_t and_;
      extern const operator_t or_;
      extern const operator_t bitAnd;
      extern const operator_t bitOr;
      extern const operator_t xor_;
      extern const operator_t leftShift;
      extern const operator_t rightShift;

      extern const operator_t assign;
      extern const operator_t addEq;
      extern const operator_t subEq;
      extern const operator_t multEq;
      extern const operator_t divEq;
      extern const operator_t modEq;
      extern const operator_t andEq;
      extern const operator_t orEq;
      extern const operator_t xorEq;
      extern const operator_t leftShiftEq;
      extern const operator_t rightShiftEq;

      extern const operator_t comma;

      // Non-Overloadable
      extern const operator_t scope;
      extern const operator_t dot;
      extern const operator_t dotStar;
      extern const operator_t arrow;
      extern const operator_t arrowStar;
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

      //---[ Special ]------------------
      extern const operator_t hash;
      extern const operator_t hashhash;

      extern const operator_t semicolon;
      extern const operator_t ellipsis;
      //================================
    }
  }
}

#endif
