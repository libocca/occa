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

    class operatorType {
    public:
      static const optype_t not_              = (1L << 0);
      static const optype_t positive          = (1L << 1);
      static const optype_t negative          = (1L << 2);
      static const optype_t tilde             = (1L << 3);
      static const optype_t leftIncrement     = (1L << 4);
      static const optype_t rightIncrement    = (1L << 5);
      static const optype_t increment         = (leftIncrement |
                                                 rightIncrement);
      static const optype_t leftDecrement     = (1L << 6);
      static const optype_t rightDecrement    = (1L << 7);
      static const optype_t decrement         = (leftDecrement |
                                                 rightDecrement);

      static const optype_t add               = (1L << 8);
      static const optype_t sub               = (1L << 9);
      static const optype_t mult              = (1L << 10);
      static const optype_t div               = (1L << 11);
      static const optype_t mod               = (1L << 12);
      static const optype_t arithmetic        = (add  |
                                                 sub  |
                                                 mult |
                                                 div  |
                                                 mod);

      static const optype_t lessThan          = (1L << 13);
      static const optype_t lessThanEq        = (1L << 14);
      static const optype_t equal             = (1L << 15);
      static const optype_t notEqual          = (1L << 16);
      static const optype_t greaterThan       = (1L << 17);
      static const optype_t greaterThanEq     = (1L << 18);
      static const optype_t comparison        = (lessThan    |
                                                 lessThanEq  |
                                                 equal       |
                                                 notEqual    |
                                                 greaterThan |
                                                 greaterThanEq);

      static const optype_t and_              = (1L << 19);
      static const optype_t or_               = (1L << 20);
      static const optype_t boolean           = (and_ |
                                                 or_);

      static const optype_t bitAnd            = (1L << 21);
      static const optype_t bitOr             = (1L << 22);
      static const optype_t xor_              = (1L << 23);
      static const optype_t leftShift         = (1L << 24);
      static const optype_t rightShift        = (1L << 25);
      static const optype_t shift             = (leftShift |
                                                 rightShift);
      static const optype_t bitOp             = (bitAnd    |
                                                 bitOr     |
                                                 xor_      |
                                                 leftShift |
                                                 rightShift);

      static const optype_t assign            = (1L << 26);
      static const optype_t addEq             = (1L << 27);
      static const optype_t subEq             = (1L << 28);
      static const optype_t multEq            = (1L << 29);
      static const optype_t divEq             = (1L << 30);
      static const optype_t modEq             = (1L << 31);
      static const optype_t andEq             = (1L << 32);
      static const optype_t orEq              = (1L << 33);
      static const optype_t xorEq             = (1L << 34);
      static const optype_t leftShiftEq       = (1L << 35);
      static const optype_t rightShiftEq      = (1L << 36);
      static const optype_t assignment        = (assign      |
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

      static const optype_t comma             = (1L << 37);
      static const optype_t scope             = (1L << 38);
      static const optype_t dot               = (1L << 39);
      static const optype_t dotStar           = (1L << 40);
      static const optype_t arrow             = (1L << 41);
      static const optype_t arrowStar         = (1L << 42);

      static const optype_t leftUnary         = (not_          |
                                                 positive      |
                                                 negative      |
                                                 tilde         |
                                                 leftIncrement |
                                                 rightDecrement);

      static const optype_t rightUnary        = (rightIncrement |
                                                 rightDecrement);

      static const optype_t binary            = (add           |
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

      static const optype_t ternary           = (3L << 43);
      static const optype_t colon             = (1L << 44);

      static const optype_t braceStart        = (1L << 45);
      static const optype_t braceEnd          = (1L << 46);
      static const optype_t bracketStart      = (1L << 47);
      static const optype_t bracketEnd        = (1L << 48);
      static const optype_t parenthesesStart  = (1L << 49);
      static const optype_t parenthesesEnd    = (1L << 50);

      static const optype_t braces            = (braceStart       |
                                                 braceEnd);
      static const optype_t brackets          = (bracketStart     |
                                                 bracketEnd);
      static const optype_t parentheses       = (parenthesesStart |
                                                 parenthesesEnd);

      static const optype_t pair              = (braceStart       |
                                                 braceEnd         |
                                                 bracketStart     |
                                                 bracketEnd       |
                                                 parenthesesStart |
                                                 parenthesesEnd);

      static const optype_t pairStart         = (braceStart       |
                                                 bracketStart     |
                                                 parenthesesStart);

      static const optype_t pairEnd           = (braceEnd         |
                                                 bracketEnd       |
                                                 parenthesesEnd);

      static const optype_t hash              = (1L << 51);
      static const optype_t hashhash          = (1L << 52);
      static const optype_t preprocessor      = (hash |
                                                 hashhash);

      static const optype_t semicolon         = (1L << 53);
      static const optype_t ellipsis          = (1L << 54);

      static const optype_t special           = (hash           |
                                                 hashhash       |
                                                 semicolon      |
                                                 ellipsis);

      static const optype_t overloadable      = (not_           |
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
    };

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
