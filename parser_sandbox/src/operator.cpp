#include "operator.hpp"

namespace occa {
  namespace lang {
    namespace op {
      //---[ Left Unary ]---------------
      const operator_t not_           ("!"  , operatorType::not_          , 1);
      const operator_t positive       ("+"  , operatorType::positive      , 1);
      const operator_t negative       ("-"  , operatorType::negative      , 1);
      const operator_t tilde          ("~"  , operatorType::tilde         , 1);
      const operator_t leftIncrement  ("++" , operatorType::leftIncrement , 1);
      const operator_t leftDecrement  ("--" , operatorType::leftDecrement , 1);
      //================================

      //---[ Right Unary ]--------------
      const operator_t rightIncrement ("++" , operatorType::rightIncrement, 1);
      const operator_t rightDecrement ("--" , operatorType::rightDecrement, 1);
      //================================

      //---[ Binary ]-------------------
      const operator_t add            ("+"  , operatorType::add           , 1);
      const operator_t sub            ("-"  , operatorType::sub           , 1);
      const operator_t mult           ("*"  , operatorType::mult          , 1);
      const operator_t div            ("/"  , operatorType::div           , 1);
      const operator_t mod            ("%"  , operatorType::mod           , 1);
      const operator_t lessThan       ("<"  , operatorType::lessThan      , 1);
      const operator_t lessThanEq     ("<=" , operatorType::lessThanEq    , 1);
      const operator_t equal          ("==" , operatorType::equal         , 1);
      const operator_t notEqual       ("!=" , operatorType::notEqual      , 1);
      const operator_t greaterThan    (">"  , operatorType::greaterThan   , 1);
      const operator_t greaterThanEq  (">=" , operatorType::greaterThanEq , 1);
      const operator_t and_           ("&&" , operatorType::and_          , 1);
      const operator_t or_            ("||" , operatorType::or_           , 1);
      const operator_t bitAnd         ("&"  , operatorType::bitAnd        , 1);
      const operator_t bitOr          ("|"  , operatorType::bitOr         , 1);
      const operator_t xor_           ("^"  , operatorType::xor_          , 1);
      const operator_t leftShift      ("<<" , operatorType::leftShift     , 1);
      const operator_t rightShift     (">>" , operatorType::rightShift    , 1);
      const operator_t addEq          ("+=" , operatorType::addEq         , 1);
      const operator_t subEq          ("-=" , operatorType::subEq         , 1);
      const operator_t multEq         ("*=" , operatorType::multEq        , 1);
      const operator_t divEq          ("/=" , operatorType::divEq         , 1);
      const operator_t modEq          ("%=" , operatorType::modEq         , 1);
      const operator_t andEq          ("&=" , operatorType::andEq         , 1);
      const operator_t orEq           ("|=" , operatorType::orEq          , 1);
      const operator_t xorEq          ("^=" , operatorType::xorEq         , 1);
      const operator_t leftShiftEq    ("<<=", operatorType::leftShiftEq   , 1);
      const operator_t rightShiftEq   (">>=", operatorType::rightShiftEq  , 1);
      //================================

      //---[ Ternary ]------------------
      const operator_t ternary        ("?:" , operatorType::ternary       , 1);
      //================================
    }

    operator_t::operator_t(const std::string &str_,
                           optype_t optype_,
                           int precedence_) :
      str(str_),
      optype(optype_),
      precedence(precedence_) {}

    void operator_t::print(printer_t &pout) const {
      pout << str;
    }
  }
}
