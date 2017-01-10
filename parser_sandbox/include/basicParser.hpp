#ifndef OCCA_PARSER_BASICPARSER_HEADER2
#define OCCA_PARSER_BASICPARSER_HEADER2

#include <iostream>

#include "occa/types.hpp"
#include "occa/tools/lex.hpp"

#  if 0
namespace occa {
  class precedence_t;
  class operator_t;

  typedef std::vector<precedence_t> precedenceVector_t;
  typedef std::vector<operator_t>   operatorVector_t;

  enum operatorType {
    none       = (1 << 0),
    leftUnary  = (1 << 1),
    rightUnary = (1 << 2),
    binary     = (1 << 3)
  };

  class operator_t {
  public:
    operatorType type;
    std::string symbol;

    static const std::string typeCast;
    static const std::string funcCall;
    static const std::string subscript;
    static const std::string cCast;
    static const std::string sizeof_;
    static const std::string new_;
    static const std::string delete_;
    static const std::string ternary;
    static const std::string throw_;

    operator_t(const std::string &symbol_) :
      symbol(symbol_),
      type(none) {}

    operator_t(const std::string &symbol_, const operatorType type_) :
      symbol(symbol_),
      type(type_) {}

    inline const bool isSymbol() const {
      return (symbol[0] != '_');
    }
  };

  const std::string operator_t::typeCast  = "_1";
  const std::string operator_t::funcCall  = "_2";
  const std::string operator_t::subscript = "_3";
  const std::string operator_t::cCast     = "_4";
  const std::string operator_t::sizeof_   = "_5";
  const std::string operator_t::new_      = "_6";
  const std::string operator_t::delete_   = "_7";
  const std::string operator_t::ternary   = "_8";
  const std::string operator_t::throw_    = "_9";

  enum precedenceType {
    leftToRight, rightToLeft
  };

  class precedence_t {
  public:
    precedenceType type;
    std::vector<operator_t> ops;

    precedence_t(precedenceType type_) :
      type(type_) {}
  };

  const precedenceVector_t& getOperatorPrecendence() {
    static precedenceVector_t precedence;
    if (precedence.size() == 0) {
      precedence.push_back(precedence_t(leftToRight));
      precedence[ 0].ops.push_back(operator_t("::" , binary));

      precedence.push_back(precedence_t(rightToLeft));
      precedence[ 1].ops.push_back(operator_t("++"  , rightUnary));
      precedence[ 1].ops.push_back(operator_t("--"  , rightUnary));
      precedence[ 1].ops.push_back(operator_t(operator_t::typeCast));
      precedence[ 1].ops.push_back(operator_t(operator_t::funcCall));
      precedence[ 1].ops.push_back(operator_t(operator_t::subscript));
      precedence[ 1].ops.push_back(operator_t("."   , binary));
      precedence[ 1].ops.push_back(operator_t("->"  , binary));

      precedence.push_back(precedence_t(leftToRight));
      precedence[ 2].ops.push_back(operator_t("++"  , leftUnary));
      precedence[ 2].ops.push_back(operator_t("--"  , leftUnary));
      precedence[ 2].ops.push_back(operator_t("+"   , leftUnary));
      precedence[ 2].ops.push_back(operator_t("-"   , leftUnary));
      precedence[ 2].ops.push_back(operator_t("!"   , leftUnary));
      precedence[ 2].ops.push_back(operator_t("~"   , leftUnary));
      precedence[ 2].ops.push_back(operator_t(operator_t::cCast));
      precedence[ 2].ops.push_back(operator_t("*"   , leftUnary));
      precedence[ 2].ops.push_back(operator_t("&"   , leftUnary));
      precedence[ 2].ops.push_back(operator_t(operator_t::sizeof_));
      precedence[ 2].ops.push_back(operator_t(operator_t::new_));
      precedence[ 2].ops.push_back(operator_t(operator_t::delete_));

      precedence.push_back(precedence_t(leftToRight));
      precedence[ 3].ops.push_back(operator_t(".*"  , binary));
      precedence[ 3].ops.push_back(operator_t("->*" , binary));

      precedence.push_back(precedence_t(leftToRight));
      precedence[ 4].ops.push_back(operator_t("*"   , binary));
      precedence[ 4].ops.push_back(operator_t("/"   , binary));
      precedence[ 4].ops.push_back(operator_t("%"   , binary));

      precedence.push_back(precedence_t(leftToRight));
      precedence[ 5].ops.push_back(operator_t("+"   , binary));
      precedence[ 5].ops.push_back(operator_t("-"   , binary));

      precedence.push_back(precedence_t(leftToRight));
      precedence[ 6].ops.push_back(operator_t("<<"  , binary));
      precedence[ 6].ops.push_back(operator_t(">>"  , binary));

      precedence.push_back(precedence_t(leftToRight));
      precedence[ 7].ops.push_back(operator_t("<"   , binary));
      precedence[ 7].ops.push_back(operator_t("<="  , binary));
      precedence[ 7].ops.push_back(operator_t(">="  , binary));
      precedence[ 7].ops.push_back(operator_t(">"   , binary));

      precedence.push_back(precedence_t(leftToRight));
      precedence[ 8].ops.push_back(operator_t("=="  , binary));
      precedence[ 8].ops.push_back(operator_t("!="  , binary));

      precedence.push_back(precedence_t(leftToRight));
      precedence[ 9].ops.push_back(operator_t("&"   , binary));

      precedence.push_back(precedence_t(leftToRight));
      precedence[10].ops.push_back(operator_t("^"   , binary));

      precedence.push_back(precedence_t(leftToRight));
      precedence[11].ops.push_back(operator_t("|"   , binary));

      precedence.push_back(precedence_t(leftToRight));
      precedence[12].ops.push_back(operator_t("&&"  , binary));

      precedence.push_back(precedence_t(leftToRight));
      precedence[13].ops.push_back(operator_t("||"  , binary));

      precedence.push_back(precedence_t(rightToLeft));
      precedence[14].ops.push_back(operator_t(operator_t::ternary));
      precedence[14].ops.push_back(operator_t(operator_t::throw_));
      precedence[14].ops.push_back(operator_t("="   , binary));
      precedence[14].ops.push_back(operator_t("+="  , binary));
      precedence[14].ops.push_back(operator_t("-="  , binary));
      precedence[14].ops.push_back(operator_t("*="  , binary));
      precedence[14].ops.push_back(operator_t("/="  , binary));
      precedence[14].ops.push_back(operator_t("%="  , binary));
      precedence[14].ops.push_back(operator_t("<<=" , binary));
      precedence[14].ops.push_back(operator_t(">>=" , binary));
      precedence[14].ops.push_back(operator_t("&="  , binary));
      precedence[14].ops.push_back(operator_t("^="  , binary));
      precedence[14].ops.push_back(operator_t("|="  , binary));

      precedence.push_back(precedence_t(leftToRight));
      precedence[15].ops.push_back(operator_t(","   , binary));
    }
    return precedence;
  }

  const trie_t& getOperatorTrie() {
    static trie_t trie;
    if (trie.empty()) {
      const precedenceVector_t &prec = getOperatorPrecendence();
      const int levels = (int) prec.size();
      for (int level = 0; level < levels; ++level) {
        const operatorVector_t &ops = prec[level].ops;
        const int opCount = (int) ops.size();
        for (int op = 0; op < opCount; ++op) {
          if (ops[op].isSymbol()) {
            trie.add(ops[op].symbol);
          }
        }
      }
      trie.freeze();
    }
    return trie;
  }

  bool isOperator(const char c) {
    return getOperatorTrie().has(c);
  }

  const char* getOperator(const char *c) {
    return getOperatorTrie().get(c);
  }

  token_t getNextValue(const char *&c) {
    token_t token = getNextToken(c);

    if (token.type & operator_) {
      operatorToken_t &op = *((operatorToken_t*) &token);
      if (op.type & unary) {
        op.setRight(getNextToken(c));
        return op;
      } else if (op.type & (left | parentheses)) {
        return getParenthesesValue(c);
      } else {
        printError("Error processing");
        return errorToken;
      }
    }

    if (token.type & (variable | constant)) {
      return token;
    }
    return errorToken;
  }

  token_t getParenthesesValue(c) {
    // Go until )
  }

  token_t parse(char *&c, const int minPrecedence) {
    token_t left = getNextValue(c);
    if (left.type & error) {
      return errorToken;
    }

    while (true) {
      const char *cStart = c;
      operator_t op = getNextOperator(c);
      if (op.type & error) {
        return errorToken;
      }

      if ((op.type == none)   ||
          !(op.type & binary) ||
          op.precendence < minPrecendence) {
        c = cStart;
        break;
      }

      token_t right = parse(c, (op.precedence +
                                (op.associativity == leftToRight)));
      if (right.type & error) {
        return errorToken;
      }

      left = op.call(left, right);
    }
    return left;
  }
}
#  endif
#endif
