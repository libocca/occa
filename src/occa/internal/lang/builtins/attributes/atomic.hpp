#ifndef OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_ATOMIC_HEADER
#define OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_ATOMIC_HEADER

#include <functional>

#include <occa/internal/lang/attribute.hpp>

namespace occa {
  namespace lang {
    class expressionStatement;
    class blockStatement;
    class expressionStatement;

    typedef std::function<bool (blockStatement &blockSmnt)>     blockSmntBoolCallback;
    typedef std::function<bool (expressionStatement &exprSmnt)> exprSmntBoolCallback;

    namespace attributes {
      class atomic : public attribute_t {
      public:
        atomic();

        virtual const std::string& name() const;

        virtual bool forStatementType(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;

        static bool applyCodeTransformation(
          blockStatement &root,
          blockSmntBoolCallback transformBlockSmnt,
          exprSmntBoolCallback transformBasicExprSmnt
        );

        static bool applyExpressionCodeTransformation(
          expressionStatement &exprSmnt,
          blockSmntBoolCallback transformBlockSmnt,
          exprSmntBoolCallback transformBasicExprSmnt
        );

        static bool applyBlockCodeTransformation(
          blockStatement &blockSmnt,
          blockSmntBoolCallback transformBlockSmnt,
          exprSmntBoolCallback transformBasicExprSmnt
        );

        static bool isBasicExpression(expressionStatement &exprSmnt);
        static bool isMinMaxExpression(expressionStatement &exprSmnt,
                                       std::string &functionName,
                                       exprNode *&argument);
      };
    }
  }
}

#endif
