#ifndef OCCA_INTERNAL_LANG_OPERATOR_HEADER
#define OCCA_INTERNAL_LANG_OPERATOR_HEADER

#include <occa/types.hpp>
#include <occa/internal/utils/trie.hpp>
#include <occa/types/primitive.hpp>
#include <occa/internal/lang/printer.hpp>

namespace occa {
  namespace lang {
    class operator_t;

    typedef udim_t   rawOpType_t;
    typedef bitfield opType_t;

    typedef trie<const operator_t*> operatorTrie;

    namespace rawOperatorType {
      extern const rawOpType_t not_;
      extern const rawOpType_t positive;
      extern const rawOpType_t negative;
      extern const rawOpType_t tilde;
      extern const rawOpType_t leftIncrement;
      extern const rawOpType_t rightIncrement;
      extern const rawOpType_t leftDecrement;
      extern const rawOpType_t rightDecrement;

      extern const rawOpType_t add;
      extern const rawOpType_t sub;
      extern const rawOpType_t mult;
      extern const rawOpType_t div;
      extern const rawOpType_t mod;

      extern const rawOpType_t lessThan;
      extern const rawOpType_t lessThanEq;
      extern const rawOpType_t equal;
      extern const rawOpType_t compare;
      extern const rawOpType_t notEqual;
      extern const rawOpType_t greaterThan;
      extern const rawOpType_t greaterThanEq;

      extern const rawOpType_t and_;
      extern const rawOpType_t or_;

      extern const rawOpType_t bitAnd;
      extern const rawOpType_t bitOr;
      extern const rawOpType_t xor_;
      extern const rawOpType_t leftShift;
      extern const rawOpType_t rightShift;

      extern const rawOpType_t assign;
      extern const rawOpType_t addEq;
      extern const rawOpType_t subEq;
      extern const rawOpType_t multEq;
      extern const rawOpType_t divEq;
      extern const rawOpType_t modEq;
      extern const rawOpType_t andEq;
      extern const rawOpType_t orEq;
      extern const rawOpType_t xorEq;
      extern const rawOpType_t leftShiftEq;
      extern const rawOpType_t rightShiftEq;

      extern const rawOpType_t comma;
      extern const rawOpType_t scope;
      extern const rawOpType_t dereference;
      extern const rawOpType_t address;
      extern const rawOpType_t dot;
      extern const rawOpType_t dotStar;
      extern const rawOpType_t arrow;
      extern const rawOpType_t arrowStar;

      extern const rawOpType_t questionMark;
      extern const rawOpType_t colon;

      extern const rawOpType_t braceStart;
      extern const rawOpType_t braceEnd;
      extern const rawOpType_t bracketStart;
      extern const rawOpType_t bracketEnd;
      extern const rawOpType_t parenthesesStart;
      extern const rawOpType_t parenthesesEnd;

      //---[ Special operators ]--------
      extern const rawOpType_t lineComment;
      extern const rawOpType_t blockCommentStart;

      extern const rawOpType_t hash;
      extern const rawOpType_t hashhash;

      extern const rawOpType_t semicolon;
      extern const rawOpType_t ellipsis;
      extern const rawOpType_t attribute;

      extern const rawOpType_t sizeof_;
      extern const rawOpType_t sizeof_pack_;
      extern const rawOpType_t new_;
      extern const rawOpType_t delete_;
      extern const rawOpType_t throw_;

      extern const rawOpType_t typeid_;
      extern const rawOpType_t noexcept_;
      extern const rawOpType_t alignof_;

      extern const rawOpType_t parenCast;

      extern const rawOpType_t cudaCallStart;
      extern const rawOpType_t cudaCallEnd;
      //================================
    }

    namespace operatorType {
      extern const opType_t none;

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
      extern const opType_t compare;
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
      extern const opType_t dereference;
      extern const opType_t address;
      extern const opType_t dot;
      extern const opType_t dotStar;
      extern const opType_t arrow;
      extern const opType_t arrowStar;

      extern const opType_t leftUnary;
      extern const opType_t rightUnary;
      extern const opType_t unary;
      extern const opType_t binary;

      extern const opType_t questionMark;
      extern const opType_t colon;
      extern const opType_t ternary;

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
      extern const opType_t comment;

      extern const opType_t hash;
      extern const opType_t hashhash;
      extern const opType_t preprocessor;

      extern const opType_t semicolon;
      extern const opType_t ellipsis;
      extern const opType_t attribute;

      extern const opType_t sizeof_;
      extern const opType_t sizeof_pack_;
      extern const opType_t new_;
      extern const opType_t delete_;
      extern const opType_t throw_;

      extern const opType_t typeid_;
      extern const opType_t noexcept_;
      extern const opType_t alignof_;

      extern const opType_t parenCast;

      extern const opType_t cudaCallStart;
      extern const opType_t cudaCallEnd;
      extern const opType_t cudaCall;

      extern const opType_t funcType;
      extern const opType_t special;

      //---[ Ambiguous Symbols ]--------
      extern const opType_t plus;
      extern const opType_t minus;
      extern const opType_t plusplus;
      extern const opType_t minusminus;
      extern const opType_t ampersand;
      extern const opType_t asterisk;
      extern const opType_t ambiguous;
      //================================

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
    };

    printer& operator << (printer &pout,
                          const operator_t &op);

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

    class pairOperator_t : public operator_t {
    public:
      std::string pairStr;

      pairOperator_t(const std::string &str_,
                     const std::string &pairStr_,
                     opType_t opType_);
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
      extern const binaryOperator_t compare;
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
      extern const unaryOperator_t  globalScope;
      extern const unaryOperator_t  dereference;
      extern const unaryOperator_t  address;
      extern const binaryOperator_t dot;
      extern const binaryOperator_t dotStar;
      extern const binaryOperator_t arrow;
      extern const binaryOperator_t arrowStar;
      //================================

      //---[ Ternary ]------------------
      extern const unaryOperator_t questionMark;
      extern const unaryOperator_t colon;
      extern const operator_t ternary;
      //================================

      //---[ Pairs ]--------------------
      extern const pairOperator_t braceStart;
      extern const pairOperator_t braceEnd;
      extern const pairOperator_t bracketStart;
      extern const pairOperator_t bracketEnd;
      extern const pairOperator_t parenthesesStart;
      extern const pairOperator_t parenthesesEnd;
      //================================

      //---[ Comments ]-----------------
      extern const operator_t lineComment;
      extern const operator_t blockCommentStart;
      //================================

      //---[ Special ]------------------
      extern const operator_t hash;
      extern const operator_t hashhash;

      extern const operator_t semicolon;
      extern const operator_t ellipsis;
      extern const operator_t attribute;

      extern const unaryOperator_t sizeof_;
      extern const unaryOperator_t sizeof_pack_;
      extern const unaryOperator_t new_;
      extern const unaryOperator_t delete_;
      extern const unaryOperator_t throw_;

      extern const unaryOperator_t typeid_;
      extern const unaryOperator_t noexcept_;
      extern const unaryOperator_t alignof_;

      extern const unaryOperator_t parenCast;

      extern const pairOperator_t cudaCallStart;
      extern const pairOperator_t cudaCallEnd;
      //================================

      //---[ Associativity ]------------
      extern const int leftAssociative;
      extern const int rightAssociative;
      extern const int associativity[19];
      //================================
    }

    void getOperators(operatorTrie &operators);
  }
}

#endif
