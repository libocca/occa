#ifndef OCCA_LANG_TRANSFORMS_BUILTINS_REPLACER_HEADER
#define OCCA_LANG_TRANSFORMS_BUILTINS_REPLACER_HEADER

#include <occa/lang/transforms/exprTransform.hpp>
#include <occa/lang/transforms/builtins/finders.hpp>

namespace occa {
  namespace lang {
    namespace transforms {
      class typeReplacer_t : public statementExprTransform {
      private:
        const type_t *from;
        type_t *to;

      public:
        typeReplacer_t();

        virtual exprNode* transformExprNode(exprNode &node);

        void set(const type_t &from_,
                 type_t &to_);

        bool applyToExpr(exprNode *&expr);
      };

      class variableReplacer_t : public statementExprTransform {
      private:
        const variable_t *from;
        variable_t *to;

      public:
        variableReplacer_t();

        virtual exprNode* transformExprNode(exprNode &node);

        void set(const variable_t &from_,
                 variable_t &to_);

        bool applyToExpr(exprNode *&expr);
      };

      class functionReplacer_t : public statementExprTransform {
      private:
        const function_t *from;
        function_t *to;

      public:
        functionReplacer_t();

        virtual exprNode* transformExprNode(exprNode &node);

        void set(const function_t &from_,
                 function_t &to_);

        bool applyToExpr(exprNode *&expr);
      };
    }

    void replaceTypes(statement_t &smnt,
                      const type_t &from,
                      type_t &to);

    void replaceTypes(exprNode &expr,
                      const type_t &from,
                      type_t &to);

    void replaceVariables(statement_t &smnt,
                          const variable_t &from,
                          variable_t &to);

    void replaceVariables(exprNode &expr,
                          const variable_t &from,
                          variable_t &to);

    void replaceFunctions(statement_t &smnt,
                          const function_t &from,
                          function_t &to);

    void replaceFunctions(exprNode &expr,
                          const function_t &from,
                          function_t &to);

    void replaceKeywords(statement_t &smnt,
                         const keyword_t &from,
                         keyword_t &to);

    void replaceKeywords(exprNode &expr,
                         const keyword_t &from,
                         keyword_t &to);
  }
}

#endif
