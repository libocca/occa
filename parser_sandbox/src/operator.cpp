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
    namespace rawOperatorType {
      const rawOpType_t not_              (1L << 1);
      const rawOpType_t positive          (1L << 2);
      const rawOpType_t negative          (1L << 3);
      const rawOpType_t tilde             (1L << 4);
      const rawOpType_t leftIncrement     (1L << 5);
      const rawOpType_t rightIncrement    (1L << 6);
      const rawOpType_t leftDecrement     (1L << 7);
      const rawOpType_t rightDecrement    (1L << 8);

      const rawOpType_t add               (1L << 9);
      const rawOpType_t sub               (1L << 10);
      const rawOpType_t mult              (1L << 11);
      const rawOpType_t div               (1L << 12);
      const rawOpType_t mod               (1L << 13);

      const rawOpType_t lessThan          (1L << 14);
      const rawOpType_t lessThanEq        (1L << 15);
      const rawOpType_t equal             (1L << 16);
      const rawOpType_t notEqual          (1L << 17);
      const rawOpType_t greaterThan       (1L << 18);
      const rawOpType_t greaterThanEq     (1L << 19);

      const rawOpType_t and_              (1L << 20);
      const rawOpType_t or_               (1L << 21);

      const rawOpType_t bitAnd            (1L << 22);
      const rawOpType_t bitOr             (1L << 23);
      const rawOpType_t xor_              (1L << 24);
      const rawOpType_t leftShift         (1L << 25);
      const rawOpType_t rightShift        (1L << 26);

      const rawOpType_t assign            (1L << 27);
      const rawOpType_t addEq             (1L << 28);
      const rawOpType_t subEq             (1L << 29);
      const rawOpType_t multEq            (1L << 30);
      const rawOpType_t divEq             (1L << 31);
      const rawOpType_t modEq             (1L << 32);
      const rawOpType_t andEq             (1L << 33);
      const rawOpType_t orEq              (1L << 34);
      const rawOpType_t xorEq             (1L << 35);
      const rawOpType_t leftShiftEq       (1L << 36);
      const rawOpType_t rightShiftEq      (1L << 37);

      const rawOpType_t comma             (1L << 38);
      const rawOpType_t scope             (1L << 39);
      const rawOpType_t dereference       (1L << 40);
      const rawOpType_t address           (1L << 41);
      const rawOpType_t dot               (1L << 42);
      const rawOpType_t dotStar           (1L << 43);
      const rawOpType_t arrow             (1L << 44);
      const rawOpType_t arrowStar         (1L << 45);

      const rawOpType_t questionMark      (1L << 46);
      const rawOpType_t colon             (1L << 47);

      // End = (Start << 1)
      const rawOpType_t braceStart        (1L << 48);
      const rawOpType_t braceEnd          (1L << 49);
      const rawOpType_t bracketStart      (1L << 50);
      const rawOpType_t bracketEnd        (1L << 51);
      const rawOpType_t parenthesesStart  (1L << 52);
      const rawOpType_t parenthesesEnd    (1L << 53);

      // Special operators
      const rawOpType_t lineComment       (1L << 0);
      const rawOpType_t blockCommentStart (1L << 1);
      const rawOpType_t blockCommentEnd   (1L << 2);

      const rawOpType_t hash              (1L << 3);
      const rawOpType_t hashhash          (1L << 4);

      const rawOpType_t semicolon         (1L << 5);
      const rawOpType_t ellipsis          (1L << 6);
      const rawOpType_t attribute         (1L << 7);
    }

    namespace operatorType {
      const opType_t not_              (0, rawOperatorType::not_);
      const opType_t positive          (0, rawOperatorType::positive);
      const opType_t negative          (0, rawOperatorType::negative);
      const opType_t tilde             (0, rawOperatorType::tilde);
      const opType_t leftIncrement     (0, rawOperatorType::leftIncrement);
      const opType_t rightIncrement    (0, rawOperatorType::rightIncrement);
      const opType_t increment         = (leftIncrement |
                                          rightIncrement);
      const opType_t leftDecrement     (0, rawOperatorType::leftDecrement);
      const opType_t rightDecrement    (0, rawOperatorType::rightDecrement);
      const opType_t decrement         = (leftDecrement |
                                          rightDecrement);

      const opType_t add               (0, rawOperatorType::add);
      const opType_t sub               (0, rawOperatorType::sub);
      const opType_t mult              (0, rawOperatorType::mult);
      const opType_t div               (0, rawOperatorType::div);
      const opType_t mod               (0, rawOperatorType::mod);
      const opType_t arithmetic        = (add  |
                                          sub  |
                                          mult |
                                          div  |
                                          mod);

      const opType_t lessThan          (0, rawOperatorType::lessThan);
      const opType_t lessThanEq        (0, rawOperatorType::lessThanEq);
      const opType_t equal             (0, rawOperatorType::equal);
      const opType_t notEqual          (0, rawOperatorType::notEqual);
      const opType_t greaterThan       (0, rawOperatorType::greaterThan);
      const opType_t greaterThanEq     (0, rawOperatorType::greaterThanEq);
      const opType_t comparison        = (lessThan    |
                                          lessThanEq  |
                                          equal       |
                                          notEqual    |
                                          greaterThan |
                                          greaterThanEq);

      const opType_t and_              (0, rawOperatorType::and_);
      const opType_t or_               (0, rawOperatorType::or_);
      const opType_t boolean           = (and_ |
                                          or_);

      const opType_t bitAnd            (0, rawOperatorType::bitAnd);
      const opType_t bitOr             (0, rawOperatorType::bitOr);
      const opType_t xor_              (0, rawOperatorType::xor_);
      const opType_t leftShift         (0, rawOperatorType::leftShift);
      const opType_t rightShift        (0, rawOperatorType::rightShift);
      const opType_t shift             = (leftShift |
                                          rightShift);
      const opType_t bitOp             = (bitAnd    |
                                          bitOr     |
                                          xor_      |
                                          leftShift |
                                          rightShift);

      const opType_t assign            (0, rawOperatorType::assign);
      const opType_t addEq             (0, rawOperatorType::addEq);
      const opType_t subEq             (0, rawOperatorType::subEq);
      const opType_t multEq            (0, rawOperatorType::multEq);
      const opType_t divEq             (0, rawOperatorType::divEq);
      const opType_t modEq             (0, rawOperatorType::modEq);
      const opType_t andEq             (0, rawOperatorType::andEq);
      const opType_t orEq              (0, rawOperatorType::orEq);
      const opType_t xorEq             (0, rawOperatorType::xorEq);
      const opType_t leftShiftEq       (0, rawOperatorType::leftShiftEq);
      const opType_t rightShiftEq      (0, rawOperatorType::rightShiftEq);
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

      const opType_t comma             (0, rawOperatorType::comma);
      const opType_t scope             (0, rawOperatorType::scope);
      const opType_t dereference       (0, rawOperatorType::dereference);
      const opType_t address           (0, rawOperatorType::address);
      const opType_t dot               (0, rawOperatorType::dot);
      const opType_t dotStar           (0, rawOperatorType::dotStar);
      const opType_t arrow             (0, rawOperatorType::arrow);
      const opType_t arrowStar         (0, rawOperatorType::arrowStar);

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

      const opType_t questionMark      (0, rawOperatorType::questionMark);
      const opType_t colon             (0, rawOperatorType::colon);
      const opType_t ternary           = (questionMark |
                                          colon);

      // End = (Start << 1)
      const opType_t braceStart        (0, rawOperatorType::braceStart);
      const opType_t braceEnd          (0, rawOperatorType::braceEnd);
      const opType_t bracketStart      (0, rawOperatorType::bracketStart);
      const opType_t bracketEnd        (0, rawOperatorType::bracketEnd);
      const opType_t parenthesesStart  (0, rawOperatorType::parenthesesStart);
      const opType_t parenthesesEnd    (0, rawOperatorType::parenthesesEnd);

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

      // Special operators
      const opType_t lineComment       (rawOperatorType::lineComment      , 0);
      const opType_t blockCommentStart (rawOperatorType::blockCommentStart, 0);
      const opType_t blockCommentEnd   (rawOperatorType::blockCommentEnd  , 0);
      const opType_t comment           = (lineComment       |
                                          blockCommentStart |
                                          blockCommentEnd);

      const opType_t hash              (rawOperatorType::hash    , 0);
      const opType_t hashhash          (rawOperatorType::hashhash, 0);
      const opType_t preprocessor      = (hash |
                                          hashhash);

      const opType_t semicolon         (rawOperatorType::semicolon, 0);
      const opType_t ellipsis          (rawOperatorType::ellipsis , 0);
      const opType_t attribute         (rawOperatorType::attribute, 0);

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
      switch(opType.b2) {
      case rawOperatorType::not_           : return not_(value);
      case rawOperatorType::positive       : return positive(value);
      case rawOperatorType::negative       : return negative(value);
      case rawOperatorType::tilde          : return tilde(value);
      case rawOperatorType::leftIncrement  : return leftIncrement(value);
      case rawOperatorType::leftDecrement  : return leftDecrement(value);

      case rawOperatorType::rightIncrement : return rightIncrement(value);
      case rawOperatorType::rightDecrement : return rightDecrement(value);
      default:
        return primitive();
      }
    }

    primitive binaryOperator_t::operator () (primitive &leftValue,
                                             primitive &rightValue) const {
      switch(opType.b2) {
      case rawOperatorType::add          : return add(leftValue, rightValue);
      case rawOperatorType::sub          : return sub(leftValue, rightValue);
      case rawOperatorType::mult         : return mult(leftValue, rightValue);
      case rawOperatorType::div          : return div(leftValue, rightValue);
      case rawOperatorType::mod          : return mod(leftValue, rightValue);

      case rawOperatorType::lessThan     : return lessThan(leftValue, rightValue);
      case rawOperatorType::lessThanEq   : return lessThanEq(leftValue, rightValue);
      case rawOperatorType::equal        : return equal(leftValue, rightValue);
      case rawOperatorType::notEqual     : return notEqual(leftValue, rightValue);
      case rawOperatorType::greaterThan  : return greaterThan(leftValue, rightValue);
      case rawOperatorType::greaterThanEq: return greaterThanEq(leftValue, rightValue);

      case rawOperatorType::and_         : return and_(leftValue, rightValue);
      case rawOperatorType::or_          : return or_(leftValue, rightValue);
      case rawOperatorType::bitAnd       : return bitAnd(leftValue, rightValue);
      case rawOperatorType::bitOr        : return bitOr(leftValue, rightValue);
      case rawOperatorType::xor_         : return xor_(leftValue, rightValue);
      case rawOperatorType::leftShift    : return leftShift(leftValue, rightValue);
      case rawOperatorType::rightShift   : return rightShift(leftValue, rightValue);

      case rawOperatorType::assign       : return assign(leftValue, rightValue);
      case rawOperatorType::addEq        : return addEq(leftValue, rightValue);
      case rawOperatorType::subEq        : return subEq(leftValue, rightValue);
      case rawOperatorType::multEq       : return multEq(leftValue, rightValue);
      case rawOperatorType::divEq        : return divEq(leftValue, rightValue);
      case rawOperatorType::modEq        : return modEq(leftValue, rightValue);
      case rawOperatorType::andEq        : return bitAndEq(leftValue, rightValue);
      case rawOperatorType::orEq         : return bitOrEq(leftValue, rightValue);
      case rawOperatorType::xorEq        : return xorEq(leftValue, rightValue);
      case rawOperatorType::leftShiftEq  : return leftShiftEq(leftValue, rightValue);
      case rawOperatorType::rightShiftEq : return rightShiftEq(leftValue, rightValue);

      case rawOperatorType::comma        : return rightValue;
      default:
        return primitive();
      }
    }
  }
}
