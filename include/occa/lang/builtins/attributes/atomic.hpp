#ifndef OCCA_LANG_BUILTINS_ATTRIBUTES_ATOMIC_HEADER
#define OCCA_LANG_BUILTINS_ATTRIBUTES_ATOMIC_HEADER

#include <functional>

#include <occa/lang/attribute.hpp>

namespace occa {
  namespace lang {
    class expressionStatement;
    class blockStatement;
    class expressionStatement;

    typedef std::function<void (blockStatement &blockSmnt)>     blockSmntVoidCallback;
    typedef std::function<void (expressionStatement &exprSmnt)> exprSmntVoidCallback;

    namespace attributes {
      class atomic : public attribute_t {
      public:
        atomic();

        virtual const std::string& name() const;

        virtual bool forStatementType(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;

        static void applyCodeTransformation(
          blockStatement &root,
          blockSmntVoidCallback transformBlockSmnt,
          exprSmntVoidCallback transformBasicExprSmnt
        );

        static void applyExpressionCodeTransformation(
          expressionStatement &exprSmnt,
          blockSmntVoidCallback transformBlockSmnt,
          exprSmntVoidCallback transformBasicExprSmnt
        );

        static void applyBlockCodeTransformation(
          blockStatement &blockSmnt,
          blockSmntVoidCallback transformBlockSmnt,
          exprSmntVoidCallback transformBasicExprSmnt
        );

        static bool isBasicExpression(expressionStatement &exprSmnt);
      };
    }
  }
}

#endif
