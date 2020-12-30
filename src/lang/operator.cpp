#include <occa/internal/lang/operator.hpp>

namespace occa {
  namespace lang {
    namespace rawOperatorType {
      const rawOpType_t not_              (((uint64_t) 1) << 1);
      const rawOpType_t positive          (((uint64_t) 1) << 2);
      const rawOpType_t negative          (((uint64_t) 1) << 3);
      const rawOpType_t tilde             (((uint64_t) 1) << 4);
      const rawOpType_t leftIncrement     (((uint64_t) 1) << 5);
      const rawOpType_t rightIncrement    (((uint64_t) 1) << 6);
      const rawOpType_t leftDecrement     (((uint64_t) 1) << 7);
      const rawOpType_t rightDecrement    (((uint64_t) 1) << 8);

      const rawOpType_t add               (((uint64_t) 1) << 9);
      const rawOpType_t sub               (((uint64_t) 1) << 10);
      const rawOpType_t mult              (((uint64_t) 1) << 11);
      const rawOpType_t div               (((uint64_t) 1) << 12);
      const rawOpType_t mod               (((uint64_t) 1) << 13);

      const rawOpType_t lessThan          (((uint64_t) 1) << 14);
      const rawOpType_t lessThanEq        (((uint64_t) 1) << 15);
      const rawOpType_t equal             (((uint64_t) 1) << 16);
      const rawOpType_t compare           (((uint64_t) 1) << 17);
      const rawOpType_t notEqual          (((uint64_t) 1) << 18);
      const rawOpType_t greaterThan       (((uint64_t) 1) << 19);
      const rawOpType_t greaterThanEq     (((uint64_t) 1) << 20);

      const rawOpType_t and_              (((uint64_t) 1) << 21);
      const rawOpType_t or_               (((uint64_t) 1) << 22);

      const rawOpType_t bitAnd            (((uint64_t) 1) << 23);
      const rawOpType_t bitOr             (((uint64_t) 1) << 24);
      const rawOpType_t xor_              (((uint64_t) 1) << 25);
      const rawOpType_t leftShift         (((uint64_t) 1) << 26);
      const rawOpType_t rightShift        (((uint64_t) 1) << 27);

      const rawOpType_t assign            (((uint64_t) 1) << 28);
      const rawOpType_t addEq             (((uint64_t) 1) << 29);
      const rawOpType_t subEq             (((uint64_t) 1) << 30);
      const rawOpType_t multEq            (((uint64_t) 1) << 31);
      const rawOpType_t divEq             (((uint64_t) 1) << 32);
      const rawOpType_t modEq             (((uint64_t) 1) << 33);
      const rawOpType_t andEq             (((uint64_t) 1) << 34);
      const rawOpType_t orEq              (((uint64_t) 1) << 35);
      const rawOpType_t xorEq             (((uint64_t) 1) << 36);
      const rawOpType_t leftShiftEq       (((uint64_t) 1) << 37);
      const rawOpType_t rightShiftEq      (((uint64_t) 1) << 38);

      const rawOpType_t comma             (((uint64_t) 1) << 39);
      const rawOpType_t scope             (((uint64_t) 1) << 40);
      const rawOpType_t globalScope       (((uint64_t) 1) << 41);
      const rawOpType_t dereference       (((uint64_t) 1) << 42);
      const rawOpType_t address           (((uint64_t) 1) << 43);
      const rawOpType_t dot               (((uint64_t) 1) << 44);
      const rawOpType_t dotStar           (((uint64_t) 1) << 45);
      const rawOpType_t arrow             (((uint64_t) 1) << 46);
      const rawOpType_t arrowStar         (((uint64_t) 1) << 47);

      const rawOpType_t questionMark      (((uint64_t) 1) << 48);
      const rawOpType_t colon             (((uint64_t) 1) << 49);
      const rawOpType_t ternary           (((uint64_t) 1) << 50);

      // End = (Start << 51)
      const rawOpType_t braceStart        (((uint64_t) 1) << 52);
      const rawOpType_t braceEnd          (((uint64_t) 1) << 53);
      const rawOpType_t bracketStart      (((uint64_t) 1) << 54);
      const rawOpType_t bracketEnd        (((uint64_t) 1) << 55);
      const rawOpType_t parenthesesStart  (((uint64_t) 1) << 56);
      const rawOpType_t parenthesesEnd    (((uint64_t) 1) << 57);

      //---[ Special operators ]--------
      const rawOpType_t lineComment       (((uint64_t) 1) << 0);
      const rawOpType_t blockCommentStart (((uint64_t) 1) << 1);

      const rawOpType_t hash              (((uint64_t) 1) << 2);
      const rawOpType_t hashhash          (((uint64_t) 1) << 3);

      const rawOpType_t semicolon         (((uint64_t) 1) << 4);
      const rawOpType_t ellipsis          (((uint64_t) 1) << 5);
      const rawOpType_t attribute         (((uint64_t) 1) << 6);

      const rawOpType_t sizeof_           (((uint64_t) 1) << 7);
      const rawOpType_t sizeof_pack_      (((uint64_t) 1) << 8);
      const rawOpType_t new_              (((uint64_t) 1) << 9);
      const rawOpType_t delete_           (((uint64_t) 1) << 10);
      const rawOpType_t throw_            (((uint64_t) 1) << 11);

      const rawOpType_t typeid_           (((uint64_t) 1) << 12);
      const rawOpType_t noexcept_         (((uint64_t) 1) << 13);
      const rawOpType_t alignof_          (((uint64_t) 1) << 14);

      const rawOpType_t parenCast         (((uint64_t) 1) << 15);

      const rawOpType_t cudaCallStart     (((uint64_t) 1) << 16);
      const rawOpType_t cudaCallEnd       (((uint64_t) 1) << 17);
      //================================
    }

    namespace operatorType {
      const opType_t none              (0, 1);

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
      const opType_t compare           (0, rawOperatorType::compare);
      const opType_t notEqual          (0, rawOperatorType::notEqual);
      const opType_t greaterThan       (0, rawOperatorType::greaterThan);
      const opType_t greaterThanEq     (0, rawOperatorType::greaterThanEq);
      const opType_t comparison        = (lessThan    |
                                          lessThanEq  |
                                          equal       |
                                          compare     |
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
      const opType_t globalScope       (0, rawOperatorType::globalScope);
      const opType_t scope             (0, rawOperatorType::scope);
      const opType_t dereference       (0, rawOperatorType::dereference);
      const opType_t address           (0, rawOperatorType::address);
      const opType_t dot               (0, rawOperatorType::dot);
      const opType_t dotStar           (0, rawOperatorType::dotStar);
      const opType_t arrow             (0, rawOperatorType::arrow);
      const opType_t arrowStar         (0, rawOperatorType::arrowStar);

      const opType_t questionMark      (0, rawOperatorType::questionMark);
      const opType_t colon             (0, rawOperatorType::colon);

      // Special
      const opType_t sizeof_           (rawOperatorType::sizeof_     , 0);
      const opType_t sizeof_pack_      (rawOperatorType::sizeof_pack_, 0);
      const opType_t new_              (rawOperatorType::new_        , 0);
      const opType_t delete_           (rawOperatorType::delete_     , 0);
      const opType_t throw_            (rawOperatorType::throw_      , 0);

      const opType_t typeid_           (rawOperatorType::typeid_  , 0);
      const opType_t noexcept_         (rawOperatorType::noexcept_, 0);
      const opType_t alignof_          (rawOperatorType::alignof_ , 0);

      const opType_t parenCast         (rawOperatorType::parenCast, 0);

      const opType_t leftUnary         = (not_          |
                                          positive      |
                                          negative      |
                                          tilde         |
                                          leftIncrement |
                                          leftDecrement |
                                          dereference   |
                                          address       |
                                          questionMark  |
                                          colon         |
                                          globalScope   |
                                          sizeof_       |
                                          sizeof_pack_  |
                                          new_          |
                                          delete_       |
                                          throw_        |
                                          typeid_       |
                                          noexcept_     |
                                          alignof_      |
                                          parenCast);

      const opType_t rightUnary        = (rightIncrement |
                                          rightDecrement);

      const opType_t unary             = (leftUnary |
                                          rightUnary);

      const opType_t binary            = (add           |
                                          sub           |
                                          mult          |
                                          div           |
                                          mod           |

                                          lessThan      |
                                          lessThanEq    |
                                          equal         |
                                          compare       |
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

      const opType_t ternary           (0, rawOperatorType::ternary);

      // End = (Start << 1)
      const opType_t braceStart        (0, rawOperatorType::braceStart);
      const opType_t braceEnd          (0, rawOperatorType::braceEnd);
      const opType_t bracketStart      (0, rawOperatorType::bracketStart);
      const opType_t bracketEnd        (0, rawOperatorType::bracketEnd);
      const opType_t parenthesesStart  (0, rawOperatorType::parenthesesStart);
      const opType_t parenthesesEnd    (0, rawOperatorType::parenthesesEnd);
      const opType_t cudaCallStart     (rawOperatorType::cudaCallStart, 0);
      const opType_t cudaCallEnd       (rawOperatorType::cudaCallEnd  , 0);

      const opType_t braces            = (braceStart       |
                                          braceEnd);
      const opType_t brackets          = (bracketStart     |
                                          bracketEnd);
      const opType_t parentheses       = (parenthesesStart |
                                          parenthesesEnd);
      const opType_t cudaCall          = (cudaCallStart    |
                                          cudaCallEnd);

      const opType_t pair              = (braceStart       |
                                          braceEnd         |
                                          bracketStart     |
                                          bracketEnd       |
                                          parenthesesStart |
                                          parenthesesEnd   |
                                          cudaCallStart    |
                                          cudaCallEnd);

      const opType_t pairStart         = (braceStart       |
                                          bracketStart     |
                                          parenthesesStart |
                                          cudaCallStart);

      const opType_t pairEnd           = (braceEnd         |
                                          bracketEnd       |
                                          parenthesesEnd   |
                                          cudaCallEnd);

      // Special operators
      const opType_t lineComment       (rawOperatorType::lineComment      , 0);
      const opType_t blockCommentStart (rawOperatorType::blockCommentStart, 0);
      const opType_t comment           = (lineComment |
                                          blockCommentStart);

      const opType_t hash              (rawOperatorType::hash    , 0);
      const opType_t hashhash          (rawOperatorType::hashhash, 0);
      const opType_t preprocessor      = (hash |
                                          hashhash);

      const opType_t semicolon         (rawOperatorType::semicolon, 0);
      const opType_t ellipsis          (rawOperatorType::ellipsis , 0);
      const opType_t attribute         (rawOperatorType::attribute, 0);

      const opType_t funcType          = (sizeof_       |
                                          sizeof_pack_  |
                                          new_          |
                                          delete_       |
                                          throw_        |
                                          typeid_       |
                                          noexcept_     |
                                          alignof_);

      const opType_t special           = (hash          |
                                          hashhash      |
                                          semicolon     |
                                          ellipsis      |
                                          attribute     |
                                          funcType      |
                                          parenCast     |
                                          cudaCallStart |
                                          cudaCallEnd);

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
                                          asterisk  |
                                          scope);
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
                                          compare        |
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
      const unaryOperator_t not_              ("!"  , operatorType::not_              , 3);
      const unaryOperator_t positive          ("+"  , operatorType::positive          , 3);
      const unaryOperator_t negative          ("-"  , operatorType::negative          , 3);
      const unaryOperator_t tilde             ("~"  , operatorType::tilde             , 3);
      const unaryOperator_t leftIncrement     ("++" , operatorType::leftIncrement     , 3);
      const unaryOperator_t leftDecrement     ("--" , operatorType::leftDecrement     , 3);
      //================================

      //---[ Right Unary ]--------------
      const unaryOperator_t rightIncrement    ("++" , operatorType::rightIncrement    , 2);
      const unaryOperator_t rightDecrement    ("--" , operatorType::rightDecrement    , 2);
      //================================

      //---[ Binary ]-------------------
      const binaryOperator_t add              ("+"  , operatorType::add              , 6);
      const binaryOperator_t sub              ("-"  , operatorType::sub              , 6);
      const binaryOperator_t mult             ("*"  , operatorType::mult             , 5);
      const binaryOperator_t div              ("/"  , operatorType::div              , 5);
      const binaryOperator_t mod              ("%"  , operatorType::mod              , 5);

      const binaryOperator_t lessThan         ("<"  , operatorType::lessThan         , 9);
      const binaryOperator_t lessThanEq       ("<=" , operatorType::lessThanEq       , 9);
      const binaryOperator_t equal            ("==" , operatorType::equal            , 10);
      const binaryOperator_t compare          ("<=>", operatorType::compare          , 8);
      const binaryOperator_t notEqual         ("!=" , operatorType::notEqual         , 10);
      const binaryOperator_t greaterThan      (">"  , operatorType::greaterThan      , 9);
      const binaryOperator_t greaterThanEq    (">=" , operatorType::greaterThanEq    , 9);

      const binaryOperator_t and_             ("&&" , operatorType::and_             , 14);
      const binaryOperator_t or_              ("||" , operatorType::or_              , 15);
      const binaryOperator_t bitAnd           ("&"  , operatorType::bitAnd           , 11);
      const binaryOperator_t bitOr            ("|"  , operatorType::bitOr            , 13);
      const binaryOperator_t xor_             ("^"  , operatorType::xor_             , 12);
      const binaryOperator_t leftShift        ("<<" , operatorType::leftShift        , 7);
      const binaryOperator_t rightShift       (">>" , operatorType::rightShift       , 7);

      const binaryOperator_t assign           ("="  , operatorType::assign           , 17);
      const binaryOperator_t addEq            ("+=" , operatorType::addEq            , 17);
      const binaryOperator_t subEq            ("-=" , operatorType::subEq            , 17);
      const binaryOperator_t multEq           ("*=" , operatorType::multEq           , 17);
      const binaryOperator_t divEq            ("/=" , operatorType::divEq            , 17);
      const binaryOperator_t modEq            ("%=" , operatorType::modEq            , 17);
      const binaryOperator_t andEq            ("&=" , operatorType::andEq            , 17);
      const binaryOperator_t orEq             ("|=" , operatorType::orEq             , 17);
      const binaryOperator_t xorEq            ("^=" , operatorType::xorEq            , 17);
      const binaryOperator_t leftShiftEq      ("<<=", operatorType::leftShiftEq      , 17);
      const binaryOperator_t rightShiftEq     (">>=", operatorType::rightShiftEq     , 17);

      const binaryOperator_t comma            (","  , operatorType::comma            , 18);

      // Non-Overloadable
      const binaryOperator_t scope            ("::" , operatorType::scope            , 1);
      const unaryOperator_t  globalScope      ("::" , operatorType::globalScope      , 1);
      const unaryOperator_t  dereference      ("*"  , operatorType::dereference      , 3);
      const unaryOperator_t  address          ("&"  , operatorType::address          , 3);
      const binaryOperator_t dot              ("."  , operatorType::dot              , 2);
      const binaryOperator_t dotStar          (".*" , operatorType::dotStar          , 4);
      const binaryOperator_t arrow            ("->" , operatorType::arrow            , 2);
      const binaryOperator_t arrowStar        ("->*", operatorType::arrowStar        , 4);
      //================================

      //---[ Ternary ]------------------
      const unaryOperator_t questionMark      ("?"  , operatorType::questionMark     , 16);
      const unaryOperator_t colon             (":"  , operatorType::colon            , 16);
      const operator_t ternary                ("?:" , operatorType::ternary          , 16);
      //================================

      //---[ Pairs ]--------------------
      const pairOperator_t braceStart         ("{", "}", operatorType::braceStart);
      const pairOperator_t braceEnd           ("}", "{", operatorType::braceEnd);
      const pairOperator_t bracketStart       ("[", "]", operatorType::bracketStart);
      const pairOperator_t bracketEnd         ("]", "[", operatorType::bracketEnd);
      const pairOperator_t parenthesesStart   ("(", ")", operatorType::parenthesesStart);
      const pairOperator_t parenthesesEnd     (")", "(", operatorType::parenthesesEnd);
      //================================

      //---[ Comments ]-----------------
      const operator_t lineComment            ("//" , operatorType::lineComment      , 0);
      const operator_t blockCommentStart      ("/*" , operatorType::blockCommentStart, 0);
      //================================

      //---[ Special ]------------------
      const operator_t hash                   ("#"  , operatorType::hash             , 0);
      const operator_t hashhash               ("##" , operatorType::hashhash         , 0);

      const operator_t semicolon              (";"  , operatorType::semicolon        , 0);
      const operator_t ellipsis               ("...", operatorType::ellipsis         , 0);
      const operator_t attribute              ("@"  , operatorType::attribute        , 0);

      const unaryOperator_t sizeof_           ("sizeof"   , operatorType::sizeof_     , 3);
      const unaryOperator_t sizeof_pack_      ("sizeof...", operatorType::sizeof_pack_, 3);
      const unaryOperator_t new_              ("new"      , operatorType::new_        , 3);
      const unaryOperator_t delete_           ("delete"   , operatorType::delete_     , 3);
      const unaryOperator_t throw_            ("throw"    , operatorType::throw_      , 17);

      const unaryOperator_t typeid_           ("typeid"   , operatorType::typeid_     , 3);
      const unaryOperator_t noexcept_         ("noexcept" , operatorType::noexcept_   , 3);
      const unaryOperator_t alignof_          ("alignof"  , operatorType::alignof_    , 3);

      const unaryOperator_t parenCast         ("()"       , operatorType::parenCast   , 3);

      const pairOperator_t cudaCallStart      ("<<<", ">>>", operatorType::cudaCallStart);
      const pairOperator_t cudaCallEnd        (">>>", "<<<", operatorType::cudaCallEnd);
      //================================

      //---[ Associativity ]------------
      const int leftAssociative  = 0;
      const int rightAssociative = 1;
      const int associativity[19] = {
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
        leftAssociative, // 16 [?:]
        rightAssociative, // 17 [assignment, throw]
        leftAssociative  // 18 [,]
      };
      //================================
    }

    operator_t::operator_t(const std::string &str_,
                           opType_t opType_,
                           int precedence_) :
      str(str_),
      opType(opType_),
      precedence(precedence_) {}

    printer& operator << (printer &pout,
                          const operator_t &op) {
      pout << op.str;
      return pout;
    }

    unaryOperator_t::unaryOperator_t(const std::string &str_,
                                     opType_t opType_,
                                     int precedence_) :
      operator_t(str_, opType_, precedence_) {}


    binaryOperator_t::binaryOperator_t(const std::string &str_,
                                       opType_t opType_,
                                       int precedence_) :
      operator_t(str_, opType_, precedence_) {}


    pairOperator_t::pairOperator_t(const std::string &str_,
                                   const std::string &pairStr_,
                                   opType_t opType_) :
      operator_t(str_, opType_, 0),
      pairStr(pairStr_) {}


    primitive unaryOperator_t::operator () (primitive &value) const {
      switch(opType.b2) {
      case rawOperatorType::not_           : return primitive::not_(value);
      case rawOperatorType::positive       : return primitive::positive(value);
      case rawOperatorType::negative       : return primitive::negative(value);
      case rawOperatorType::tilde          : return primitive::tilde(value);
      case rawOperatorType::leftIncrement  : return primitive::leftIncrement(value);
      case rawOperatorType::leftDecrement  : return primitive::leftDecrement(value);

      case rawOperatorType::rightIncrement : return primitive::rightIncrement(value);
      case rawOperatorType::rightDecrement : return primitive::rightDecrement(value);
      default:
        return primitive();
      }
    }

    primitive binaryOperator_t::operator () (primitive &leftValue,
                                             primitive &rightValue) const {
      switch(opType.b2) {
      case rawOperatorType::add          : return primitive::add(leftValue, rightValue);
      case rawOperatorType::sub          : return primitive::sub(leftValue, rightValue);
      case rawOperatorType::mult         : return primitive::mult(leftValue, rightValue);
      case rawOperatorType::div          : return primitive::div(leftValue, rightValue);
      case rawOperatorType::mod          : return primitive::mod(leftValue, rightValue);

      case rawOperatorType::lessThan     : return primitive::lessThan(leftValue, rightValue);
      case rawOperatorType::lessThanEq   : return primitive::lessThanEq(leftValue, rightValue);
      case rawOperatorType::equal        : return primitive::equal(leftValue, rightValue);
      case rawOperatorType::compare      : return primitive::compare(leftValue, rightValue);
      case rawOperatorType::notEqual     : return primitive::notEqual(leftValue, rightValue);
      case rawOperatorType::greaterThan  : return primitive::greaterThan(leftValue, rightValue);
      case rawOperatorType::greaterThanEq: return primitive::greaterThanEq(leftValue, rightValue);

      case rawOperatorType::and_         : return primitive::and_(leftValue, rightValue);
      case rawOperatorType::or_          : return primitive::or_(leftValue, rightValue);
      case rawOperatorType::bitAnd       : return primitive::bitAnd(leftValue, rightValue);
      case rawOperatorType::bitOr        : return primitive::bitOr(leftValue, rightValue);
      case rawOperatorType::xor_         : return primitive::xor_(leftValue, rightValue);
      case rawOperatorType::leftShift    : return primitive::leftShift(leftValue, rightValue);
      case rawOperatorType::rightShift   : return primitive::rightShift(leftValue, rightValue);

      case rawOperatorType::assign       : return primitive::assign(leftValue, rightValue);
      case rawOperatorType::addEq        : return primitive::addEq(leftValue, rightValue);
      case rawOperatorType::subEq        : return primitive::subEq(leftValue, rightValue);
      case rawOperatorType::multEq       : return primitive::multEq(leftValue, rightValue);
      case rawOperatorType::divEq        : return primitive::divEq(leftValue, rightValue);
      case rawOperatorType::modEq        : return primitive::modEq(leftValue, rightValue);
      case rawOperatorType::andEq        : return primitive::bitAndEq(leftValue, rightValue);
      case rawOperatorType::orEq         : return primitive::bitOrEq(leftValue, rightValue);
      case rawOperatorType::xorEq        : return primitive::xorEq(leftValue, rightValue);
      case rawOperatorType::leftShiftEq  : return primitive::leftShiftEq(leftValue, rightValue);
      case rawOperatorType::rightShiftEq : return primitive::rightShiftEq(leftValue, rightValue);

      case rawOperatorType::comma        : return rightValue;
      default:
        return primitive();
      }
    }

    void getOperators(operatorTrie &operators) {
      operators.add(op::not_.str             , &op::not_);
      operators.add(op::tilde.str            , &op::tilde);
      operators.add(op::leftIncrement.str    , &op::leftIncrement);
      operators.add(op::leftDecrement.str    , &op::leftDecrement);

      operators.add(op::add.str              , &op::add);
      operators.add(op::sub.str              , &op::sub);
      operators.add(op::mult.str             , &op::mult);
      operators.add(op::div.str              , &op::div);
      operators.add(op::mod.str              , &op::mod);

      operators.add(op::lessThan.str         , &op::lessThan);
      operators.add(op::lessThanEq.str       , &op::lessThanEq);
      operators.add(op::equal.str            , &op::equal);
      operators.add(op::notEqual.str         , &op::notEqual);
      operators.add(op::greaterThan.str      , &op::greaterThan);
      operators.add(op::greaterThanEq.str    , &op::greaterThanEq);

      operators.add(op::and_.str             , &op::and_);
      operators.add(op::or_.str              , &op::or_);

      operators.add(op::bitAnd.str           , &op::bitAnd);
      operators.add(op::bitOr.str            , &op::bitOr);
      operators.add(op::xor_.str             , &op::xor_);
      operators.add(op::leftShift.str        , &op::leftShift);
      operators.add(op::rightShift.str       , &op::rightShift);

      operators.add(op::assign.str           , &op::assign);
      operators.add(op::addEq.str            , &op::addEq);
      operators.add(op::subEq.str            , &op::subEq);
      operators.add(op::multEq.str           , &op::multEq);
      operators.add(op::divEq.str            , &op::divEq);
      operators.add(op::modEq.str            , &op::modEq);
      operators.add(op::andEq.str            , &op::andEq);
      operators.add(op::orEq.str             , &op::orEq);
      operators.add(op::xorEq.str            , &op::xorEq);
      operators.add(op::leftShiftEq.str      , &op::leftShiftEq);
      operators.add(op::rightShiftEq.str     , &op::rightShiftEq);

      operators.add(op::comma.str            , &op::comma);
      operators.add(op::scope.str            , &op::scope);
      operators.add(op::dot.str              , &op::dot);
      operators.add(op::dotStar.str          , &op::dotStar);
      operators.add(op::arrow.str            , &op::arrow);
      operators.add(op::arrowStar.str        , &op::arrowStar);
      operators.add(op::questionMark.str     , &op::questionMark);
      operators.add(op::colon.str            , &op::colon);

      operators.add(op::braceStart.str       , &op::braceStart);
      operators.add(op::braceEnd.str         , &op::braceEnd);
      operators.add(op::bracketStart.str     , &op::bracketStart);
      operators.add(op::bracketEnd.str       , &op::bracketEnd);
      operators.add(op::parenthesesStart.str , &op::parenthesesStart);
      operators.add(op::parenthesesEnd.str   , &op::parenthesesEnd);

      operators.add(op::lineComment.str      , &op::lineComment);
      operators.add(op::blockCommentStart.str, &op::blockCommentStart);

      operators.add(op::hash.str             , &op::hash);
      operators.add(op::hashhash.str         , &op::hashhash);

      operators.add(op::semicolon.str        , &op::semicolon);
      operators.add(op::ellipsis.str         , &op::ellipsis);
      operators.add(op::attribute.str        , &op::attribute);

      operators.add(op::sizeof_.str          , &op::sizeof_);
      operators.add(op::sizeof_pack_.str     , &op::sizeof_pack_);
      operators.add(op::new_.str             , &op::new_);
      operators.add(op::delete_.str          , &op::delete_);
      operators.add(op::throw_.str           , &op::throw_);

      operators.add(op::typeid_.str          , &op::typeid_);
      operators.add(op::noexcept_.str        , &op::noexcept_);
      operators.add(op::alignof_.str         , &op::alignof_);

      // Don't add parenCast since it's not an operator

      operators.add(op::cudaCallStart.str    , &op::cudaCallStart);
      operators.add(op::cudaCallEnd.str      , &op::cudaCallEnd);
    }
  }
}
