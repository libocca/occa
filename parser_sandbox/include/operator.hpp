#ifndef OCCA_PARSER_OPERATOR_HEADER2
#define OCCA_PARSER_OPERATOR_HEADER2

#include "occa/types.hpp"
#include "printer.hpp"

namespace occa {
  namespace lang {
    typedef uint64_t optype_t;

    class operatorType {
    public:
      static const optype_t not_             = (1L << 0);
      static const optype_t positive         = (1L << 1);
      static const optype_t negative         = (1L << 2);
      static const optype_t tilde            = (1L << 3);
      static const optype_t leftIncrement    = (1L << 4);
      static const optype_t rightIncrement   = (1L << 5);
      static const optype_t increment        = (leftIncrement |
                                                rightIncrement);
      static const optype_t leftDecrement    = (1L << 6);
      static const optype_t rightDecrement   = (1L << 7);
      static const optype_t decrement        = (leftDecrement |
                                                rightDecrement);

      static const optype_t add              = (1L << 8);
      static const optype_t sub              = (1L << 9);
      static const optype_t mult             = (1L << 10);
      static const optype_t div              = (1L << 11);
      static const optype_t mod              = (1L << 12);
      static const optype_t arithmetic       = (add  |
                                                sub  |
                                                mult |
                                                div |
                                                mod);

      static const optype_t lessThan         = (1L << 13);
      static const optype_t lessThanEq       = (1L << 14);
      static const optype_t equal            = (1L << 15);
      static const optype_t notEqual         = (1L << 16);
      static const optype_t greaterThan      = (1L << 17);
      static const optype_t greaterThanEq    = (1L << 18);
      static const optype_t comparison       = (lessThan    |
                                                lessThanEq  |
                                                equal       |
                                                notEqual    |
                                                greaterThan |
                                                greaterThanEq);

      static const optype_t and_             = (1L << 19);
      static const optype_t or_              = (1L << 20);
      static const optype_t boolean          = (and_ |
                                                or_);

      static const optype_t bitAnd           = (1L << 21);
      static const optype_t bitOr            = (1L << 22);
      static const optype_t xor_             = (1L << 23);
      static const optype_t leftShift        = (1L << 24);
      static const optype_t rightShift       = (1L << 25);
      static const optype_t shift            = (leftShift |
                                                rightShift);
      static const optype_t bitOp            = (bitAnd    |
                                                bitOr     |
                                                xor_      |
                                                leftShift |
                                                rightShift);

      static const optype_t addEq            = (1L << 26);
      static const optype_t subEq            = (1L << 27);
      static const optype_t multEq           = (1L << 28);
      static const optype_t divEq            = (1L << 29);
      static const optype_t modEq            = (1L << 30);
      static const optype_t andEq            = (1L << 31);
      static const optype_t orEq             = (1L << 32);
      static const optype_t xorEq            = (1L << 33);
      static const optype_t leftShiftEq      = (1L << 34);
      static const optype_t rightShiftEq     = (1L << 35);
      static const optype_t assignment       = (addEq       |
                                                subEq       |
                                                multEq      |
                                                divEq       |
                                                modEq       |
                                                andEq       |
                                                orEq        |
                                                xorEq       |
                                                leftShiftEq |
                                                rightShiftEq);

      static const optype_t leftUnary        = (not_          |
                                                positive      |
                                                negative      |
                                                tilde         |
                                                leftIncrement |
                                                rightDecrement);

      static const optype_t rightUnary       = (rightIncrement |
                                                rightDecrement);

      static const optype_t binary           = (add           |
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
                                                addEq         |
                                                subEq         |
                                                multEq        |
                                                divEq         |
                                                modEq         |
                                                andEq         |
                                                orEq          |
                                                xorEq         |
                                                leftShiftEq   |
                                                rightShiftEq);

      static const optype_t ternary          = (1L << 26);
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
      //================================

      //---[ Ternary ]------------------
      extern const operator_t ternary;
      //================================
    }
  }
}

#endif
