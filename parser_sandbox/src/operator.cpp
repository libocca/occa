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
#include "operator.hpp"

namespace occa {
  namespace lang {
    namespace operatorType {
      const opType_t not_              = (1L << 0);
      const opType_t positive          = (1L << 1);
      const opType_t negative          = (1L << 2);
      const opType_t tilde             = (1L << 3);
      const opType_t leftIncrement     = (1L << 4);
      const opType_t rightIncrement    = (1L << 5);
      const opType_t increment         = (leftIncrement |
                                          rightIncrement);
      const opType_t leftDecrement     = (1L << 6);
      const opType_t rightDecrement    = (1L << 7);
      const opType_t decrement         = (leftDecrement |
                                          rightDecrement);

      const opType_t add               = (1L << 8);
      const opType_t sub               = (1L << 9);
      const opType_t mult              = (1L << 10);
      const opType_t div               = (1L << 11);
      const opType_t mod               = (1L << 12);
      const opType_t arithmetic        = (add  |
                                          sub  |
                                          mult |
                                          div  |
                                          mod);

      const opType_t lessThan          = (1L << 13);
      const opType_t lessThanEq        = (1L << 14);
      const opType_t equal             = (1L << 15);
      const opType_t notEqual          = (1L << 16);
      const opType_t greaterThan       = (1L << 17);
      const opType_t greaterThanEq     = (1L << 18);
      const opType_t comparison        = (lessThan    |
                                          lessThanEq  |
                                          equal       |
                                          notEqual    |
                                          greaterThan |
                                          greaterThanEq);

      const opType_t and_              = (1L << 19);
      const opType_t or_               = (1L << 20);
      const opType_t boolean           = (and_ |
                                          or_);

      const opType_t bitAnd            = (1L << 21);
      const opType_t bitOr             = (1L << 22);
      const opType_t xor_              = (1L << 23);
      const opType_t leftShift         = (1L << 24);
      const opType_t rightShift        = (1L << 25);
      const opType_t shift             = (leftShift |
                                          rightShift);
      const opType_t bitOp             = (bitAnd    |
                                          bitOr     |
                                          xor_      |
                                          leftShift |
                                          rightShift);

      const opType_t assign            = (1L << 26);
      const opType_t addEq             = (1L << 27);
      const opType_t subEq             = (1L << 28);
      const opType_t multEq            = (1L << 29);
      const opType_t divEq             = (1L << 30);
      const opType_t modEq             = (1L << 31);
      const opType_t andEq             = (1L << 32);
      const opType_t orEq              = (1L << 33);
      const opType_t xorEq             = (1L << 34);
      const opType_t leftShiftEq       = (1L << 35);
      const opType_t rightShiftEq      = (1L << 36);
      const opType_t assignment        = (assign      |
                                          addEq       |
                                          subEq       |
                                          multEq      |
                                          divEq       |
                                          modEq       |
                                          andEq       |
                                          orEq        |
                                          xorEq       |
                                          leftShiftEq |
                                          rightShiftEq);

      const opType_t comma             = (1L << 37);
      const opType_t scope             = (1L << 38);
      const opType_t dereference       = (1L << 39);
      const opType_t address           = (1L << 40);
      const opType_t dot               = (1L << 41);
      const opType_t dotStar           = (1L << 42);
      const opType_t arrow             = (1L << 43);
      const opType_t arrowStar         = (1L << 44);

      const opType_t leftUnary         = (not_          |
                                          positive      |
                                          negative      |
                                          tilde         |
                                          leftIncrement |
                                          leftDecrement |
                                          dereference   |
                                          address);

      const opType_t rightUnary        = (rightIncrement |
                                          rightDecrement);

      const opType_t binary            = (add           |
                                          sub           |
                                          mult          |
                                          div           |
                                          mod           |

                                          lessThan      |
                                          lessThanEq    |
                                          equal         |
                                          notEqual      |
                                          greaterThan   |
                                          greaterThanEq |

                                          and_          |
                                          or_           |
                                          bitAnd        |
                                          bitOr         |
                                          xor_          |
                                          leftShift     |
                                          rightShift    |

                                          assign        |
                                          addEq         |
                                          subEq         |
                                          multEq        |
                                          divEq         |
                                          modEq         |
                                          andEq         |
                                          orEq          |
                                          xorEq         |

                                          leftShiftEq   |
                                          rightShiftEq  |

                                          comma         |
                                          scope         |
                                          dot           |
                                          dotStar       |
                                          arrow         |
                                          arrowStar);

      const opType_t ternary           = (3L << 45);
      const opType_t colon             = (1L << 46);

      // End = (Start << 1)
      const opType_t braceStart        = (1L << 47);
      const opType_t braceEnd          = (1L << 48);
      const opType_t bracketStart      = (1L << 49);
      const opType_t bracketEnd        = (1L << 50);
      const opType_t parenthesesStart  = (1L << 51);
      const opType_t parenthesesEnd    = (1L << 52);

      const opType_t braces            = (braceStart       |
                                          braceEnd);
      const opType_t brackets          = (bracketStart     |
                                          bracketEnd);
      const opType_t parentheses       = (parenthesesStart |
                                          parenthesesEnd);

      const opType_t pair              = (braceStart       |
                                          braceEnd         |
                                          bracketStart     |
                                          bracketEnd       |
                                          parenthesesStart |
                                          parenthesesEnd);

      const opType_t pairStart         = (braceStart       |
                                          bracketStart     |
                                          parenthesesStart);

      const opType_t pairEnd           = (braceEnd         |
                                          bracketEnd       |
                                          parenthesesEnd);

      const opType_t lineComment       = (1L << 53);
      const opType_t blockCommentStart = (1L << 54);
      const opType_t blockCommentEnd   = (1L << 55);
      const opType_t comment           = (lineComment       |
                                          blockCommentStart |
                                          blockCommentEnd);

      const opType_t hash              = (1L << 56);
      const opType_t hashhash          = (1L << 57);
      const opType_t preprocessor      = (hash |
                                          hashhash);

      const opType_t semicolon         = (1L << 58);
      const opType_t ellipsis          = (1L << 59);
      const opType_t attribute         = (1L << 60);

      const opType_t special           = (hash      |
                                          hashhash  |
                                          semicolon |
                                          ellipsis  |
                                          attribute);

      //---[ Ambiguous Symbols ]--------
      const opType_t plus              = (positive  |
                                          add);

      const opType_t minus             = (negative  |
                                          sub);

      const opType_t ampersand         = (bitAnd    |
                                          address);

      const opType_t asterisk          = (mult      |
                                          dereference);

      const opType_t ambiguous         = (plus      |
                                          minus     |
                                          increment |
                                          decrement |
                                          ampersand |
                                          asterisk);
      //================================

      const opType_t overloadable      = (not_           |
                                          positive       |
                                          negative       |
                                          tilde          |
                                          leftIncrement  |
                                          leftDecrement  |
                                          rightIncrement |
                                          rightDecrement |

                                          add            |
                                          sub            |
                                          mult           |
                                          div            |
                                          mod            |

                                          lessThan       |
                                          lessThanEq     |
                                          equal          |
                                          notEqual       |
                                          greaterThan    |
                                          greaterThanEq  |

                                          and_           |
                                          or_            |
                                          bitAnd         |
                                          bitOr          |
                                          xor_           |
                                          leftShift      |
                                          rightShift     |

                                          assign         |
                                          addEq          |
                                          subEq          |
                                          multEq         |
                                          divEq          |
                                          modEq          |
                                          andEq          |
                                          orEq           |
                                          xorEq          |
                                          leftShiftEq    |
                                          rightShiftEq   |

                                          comma);
    }

    namespace op {
      //---[ Left Unary ]---------------
      const unaryOperator_t not_           ("!"  , operatorType::not_              , 3);
      const unaryOperator_t positive       ("+"  , operatorType::positive          , 3);
      const unaryOperator_t negative       ("-"  , operatorType::negative          , 3);
      const unaryOperator_t tilde          ("~"  , operatorType::tilde             , 3);
      const unaryOperator_t leftIncrement  ("++" , operatorType::leftIncrement     , 3);
      const unaryOperator_t leftDecrement  ("--" , operatorType::leftDecrement     , 3);
      //================================

      //---[ Right Unary ]--------------
      const unaryOperator_t rightIncrement ("++" , operatorType::rightIncrement    , 2);
      const unaryOperator_t rightDecrement ("--" , operatorType::rightDecrement    , 2);
      //================================

      //---[ Binary ]-------------------
      const binaryOperator_t add            ("+"  , operatorType::add              , 6);
      const binaryOperator_t sub            ("-"  , operatorType::sub              , 6);
      const binaryOperator_t mult           ("*"  , operatorType::mult             , 5);
      const binaryOperator_t div            ("/"  , operatorType::div              , 5);
      const binaryOperator_t mod            ("%"  , operatorType::mod              , 5);

      const binaryOperator_t lessThan       ("<"  , operatorType::lessThan         , 9);
      const binaryOperator_t lessThanEq     ("<=" , operatorType::lessThanEq       , 9);
      const binaryOperator_t equal          ("==" , operatorType::equal            , 10);
      const binaryOperator_t notEqual       ("!=" , operatorType::notEqual         , 10);
      const binaryOperator_t greaterThan    (">"  , operatorType::greaterThan      , 9);
      const binaryOperator_t greaterThanEq  (">=" , operatorType::greaterThanEq    , 9);

      const binaryOperator_t and_           ("&&" , operatorType::and_             , 14);
      const binaryOperator_t or_            ("||" , operatorType::or_              , 15);
      const binaryOperator_t bitAnd         ("&"  , operatorType::bitAnd           , 11);
      const binaryOperator_t bitOr          ("|"  , operatorType::bitOr            , 13);
      const binaryOperator_t xor_           ("^"  , operatorType::xor_             , 12);
      const binaryOperator_t leftShift      ("<<" , operatorType::leftShift        , 7);
      const binaryOperator_t rightShift     (">>" , operatorType::rightShift       , 7);

      const binaryOperator_t assign         ("="  , operatorType::assign           , 16);
      const binaryOperator_t addEq          ("+=" , operatorType::addEq            , 16);
      const binaryOperator_t subEq          ("-=" , operatorType::subEq            , 16);
      const binaryOperator_t multEq         ("*=" , operatorType::multEq           , 16);
      const binaryOperator_t divEq          ("/=" , operatorType::divEq            , 16);
      const binaryOperator_t modEq          ("%=" , operatorType::modEq            , 16);
      const binaryOperator_t andEq          ("&=" , operatorType::andEq            , 16);
      const binaryOperator_t orEq           ("|=" , operatorType::orEq             , 16);
      const binaryOperator_t xorEq          ("^=" , operatorType::xorEq            , 16);
      const binaryOperator_t leftShiftEq    ("<<=", operatorType::leftShiftEq      , 16);
      const binaryOperator_t rightShiftEq   (">>=", operatorType::rightShiftEq     , 16);

      const binaryOperator_t comma          (","  , operatorType::comma            , 17);

      // Non-Overloadable
      const binaryOperator_t scope          ("::" , operatorType::scope            , 1);
      const unaryOperator_t  dereference    ("*"  , operatorType::dereference      , 3);
      const unaryOperator_t  address        ("&"  , operatorType::address          , 3);
      const binaryOperator_t dot            ("."  , operatorType::dot              , 2);
      const binaryOperator_t dotStar        (".*" , operatorType::dotStar          , 4);
      const binaryOperator_t arrow          ("->" , operatorType::arrow            , 2);
      const binaryOperator_t arrowStar      ("->*", operatorType::arrowStar        , 4);
      //================================

      //---[ Ternary ]------------------
      const operator_t ternary              ("?"  , operatorType::ternary          , 16);
      const operator_t colon                (":"  , operatorType::colon            , 16);
      //================================

      //---[ Pairs ]--------------------
      const operator_t braceStart           ("{"  , operatorType::braceStart       , 0);
      const operator_t braceEnd             ("}"  , operatorType::braceEnd         , 0);
      const operator_t bracketStart         ("["  , operatorType::bracketStart     , 0);
      const operator_t bracketEnd           ("]"  , operatorType::bracketEnd       , 0);
      const operator_t parenthesesStart     ("("  , operatorType::parenthesesStart , 0);
      const operator_t parenthesesEnd       (")"  , operatorType::parenthesesEnd   , 0);
      //================================

      //---[ Comments ]-----------------
      const operator_t lineComment          ("//" , operatorType::lineComment      , 0);
      const operator_t blockCommentStart    ("/*" , operatorType::blockCommentStart, 0);
      const operator_t blockCommentEnd      ("*/" , operatorType::blockCommentEnd  , 0);
      //================================

      //---[ Special ]------------------
      const operator_t hash                 ("#"  , operatorType::hash             , 0);
      const operator_t hashhash             ("##" , operatorType::hashhash         , 0);

      const operator_t semicolon            (";"  , operatorType::semicolon        , 0);
      const operator_t ellipsis             ("...", operatorType::ellipsis         , 0);
      const operator_t attribute            ("@"  , operatorType::attribute        , 0);
      //================================

      //---[ Associativity ]------------
      const int leftAssociative  = 0;
      const int rightAssociative = 1;
      const int associativity[18] = {
        leftAssociative,  // 0
        leftAssociative,  // 1
        leftAssociative,  // 2
        rightAssociative, // 3  [Unary operators]
        leftAssociative,  // 4
        leftAssociative,  // 5
        leftAssociative,  // 6
        leftAssociative,  // 7
        leftAssociative,  // 8
        leftAssociative,  // 9
        leftAssociative,  // 10
        leftAssociative,  // 11
        leftAssociative,  // 12
        leftAssociative,  // 13
        leftAssociative,  // 14
        leftAssociative,  // 15
        rightAssociative, // 16 [?:, throw, assignment]
        leftAssociative,  // 17
      };
      //================================
    }

    operator_t::operator_t(const std::string &str_,
                           opType_t opType_,
                           int precedence_) :
      str(str_),
      opType(opType_),
      precedence(precedence_) {}

    void operator_t::print(printer &pout) const {
      pout << str;
    }


    unaryOperator_t::unaryOperator_t(const std::string &str_,
                                     opType_t opType_,
                                     int precedence_) :
      operator_t(str_, opType_, precedence_) {}


    binaryOperator_t::binaryOperator_t(const std::string &str_,
                                       opType_t opType_,
                                       int precedence_) :
      operator_t(str_, opType_, precedence_) {}


    primitive unaryOperator_t::operator () (primitive &value) const {
      switch(opType) {
      case operatorType::not_           : return not_(value);
      case operatorType::positive       : return positive(value);
      case operatorType::negative       : return negative(value);
      case operatorType::tilde          : return tilde(value);
      case operatorType::leftIncrement  : return leftIncrement(value);
      case operatorType::leftDecrement  : return leftDecrement(value);

      case operatorType::rightIncrement : return rightIncrement(value);
      case operatorType::rightDecrement : return rightDecrement(value);
      default:
        return primitive();
      }
    }

    primitive binaryOperator_t::operator () (primitive &leftValue,
                                             primitive &rightValue) const {
      switch(opType) {
      case operatorType::add          : return add(leftValue, rightValue);
      case operatorType::sub          : return sub(leftValue, rightValue);
      case operatorType::mult         : return mult(leftValue, rightValue);
      case operatorType::div          : return div(leftValue, rightValue);
      case operatorType::mod          : return mod(leftValue, rightValue);

      case operatorType::lessThan     : return lessThan(leftValue, rightValue);
      case operatorType::lessThanEq   : return lessThanEq(leftValue, rightValue);
      case operatorType::equal        : return equal(leftValue, rightValue);
      case operatorType::notEqual     : return notEqual(leftValue, rightValue);
      case operatorType::greaterThan  : return greaterThan(leftValue, rightValue);
      case operatorType::greaterThanEq: return greaterThanEq(leftValue, rightValue);

      case operatorType::and_         : return and_(leftValue, rightValue);
      case operatorType::or_          : return or_(leftValue, rightValue);
      case operatorType::bitAnd       : return bitAnd(leftValue, rightValue);
      case operatorType::bitOr        : return bitOr(leftValue, rightValue);
      case operatorType::xor_         : return xor_(leftValue, rightValue);
      case operatorType::leftShift    : return leftShift(leftValue, rightValue);
      case operatorType::rightShift   : return rightShift(leftValue, rightValue);

      case operatorType::assign       : return assign(leftValue, rightValue);
      case operatorType::addEq        : return addEq(leftValue, rightValue);
      case operatorType::subEq        : return subEq(leftValue, rightValue);
      case operatorType::multEq       : return multEq(leftValue, rightValue);
      case operatorType::divEq        : return divEq(leftValue, rightValue);
      case operatorType::modEq        : return modEq(leftValue, rightValue);
      case operatorType::andEq        : return bitAndEq(leftValue, rightValue);
      case operatorType::orEq         : return bitOrEq(leftValue, rightValue);
      case operatorType::xorEq        : return xorEq(leftValue, rightValue);
      case operatorType::leftShiftEq  : return leftShiftEq(leftValue, rightValue);
      case operatorType::rightShiftEq : return rightShiftEq(leftValue, rightValue);

      case operatorType::comma        : return rightValue;
      default:
        return primitive();
      }
    }
  }
}
