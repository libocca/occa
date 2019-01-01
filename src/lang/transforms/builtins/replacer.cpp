#include <occa/lang/expr.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/transforms/builtins/replacer.hpp>

namespace occa {
  namespace lang {
    namespace transforms {
      typeReplacer_t::typeReplacer_t() :
        statementExprTransform(exprNodeType::type),
        from(NULL),
        to(NULL) {}

      void typeReplacer_t::set(const type_t &from_,
                               type_t &to_) {
        from = &from_;
        to   = &to_;
      }

      exprNode* typeReplacer_t::transformExprNode(exprNode &node) {
        if (!from || !to) {
          return &node;
        }
        type_t &type = ((typeNode&) node).value;
        if (&type != from) {
          return &node;
        }
        return new typeNode(node.token, *to);
      }

      variableReplacer_t::variableReplacer_t() :
        statementExprTransform(exprNodeType::variable),
        from(NULL),
        to(NULL) {}

      void variableReplacer_t::set(const variable_t &from_,
                                   variable_t &to_) {
        from = &from_;
        to   = &to_;
      }

      exprNode* variableReplacer_t::transformExprNode(exprNode &node) {
        if (!from || !to) {
          return &node;
        }
        variable_t &var = ((variableNode&) node).value;
        if (&var != from) {
          return &node;
        }
        return new variableNode(node.token, *to);
      }

      functionReplacer_t::functionReplacer_t() :
        statementExprTransform(exprNodeType::function),
        from(NULL),
        to(NULL) {}

      void functionReplacer_t::set(const function_t &from_,
                                   function_t &to_) {
        from = &from_;
        to   = &to_;
      }

      exprNode* functionReplacer_t::transformExprNode(exprNode &node) {
        if (!from || !to) {
          return &node;
        }
        function_t &func = ((functionNode&) node).value;
        if (&func != from) {
          return &node;
        }
        return new functionNode(node.token, *to);
      }
    }

    void replaceTypes(statement_t &smnt,
                      const type_t &from,
                      type_t &to) {
      transforms::typeReplacer_t replacer;
      replacer.set(from, to);
      replacer.statementTransform::apply(smnt);
    }

    void replaceTypes(exprNode &expr,
                      const type_t &from,
                      type_t &to) {
      transforms::typeReplacer_t replacer;
      replacer.set(from, to);
      replacer.exprTransform::apply(expr);
    }

    void replaceVariables(statement_t &smnt,
                          const variable_t &from,
                          variable_t &to) {
      transforms::variableReplacer_t replacer;
      replacer.set(from, to);
      replacer.statementTransform::apply(smnt);
    }

    void replaceVariables(exprNode &expr,
                          const variable_t &from,
                          variable_t &to) {
      transforms::variableReplacer_t replacer;
      replacer.set(from, to);
      replacer.exprTransform::apply(expr);
    }

    void replaceFunctions(statement_t &smnt,
                          const function_t &from,
                          function_t &to) {
      transforms::functionReplacer_t replacer;
      replacer.set(from, to);
      replacer.statementTransform::apply(smnt);
    }

    void replaceFunctions(exprNode &expr,
                          const function_t &from,
                          function_t &to) {
      transforms::functionReplacer_t replacer;
      replacer.set(from, to);
      replacer.exprTransform::apply(expr);
    }

    void replaceKeywords(statement_t &smnt,
                         const keyword_t &from,
                         keyword_t &to) {
      const int kType = from.type();
      if (kType & keywordType::type) {
        const type_t &fromValue = ((const typeKeyword&) from).type_;
        type_t &toValue = ((typeKeyword&) to).type_;
        replaceTypes(smnt, fromValue, toValue);
      }
      else if (kType & keywordType::variable) {
        const variable_t &fromValue = ((const variableKeyword&) from).variable;
        variable_t &toValue = ((variableKeyword&) to).variable;
        replaceVariables(smnt, fromValue, toValue);
      }
      else if (kType & keywordType::function) {
        const function_t &fromValue = ((const functionKeyword&) from).function;
        function_t &toValue = ((functionKeyword&) to).function;
        replaceFunctions(smnt, fromValue, toValue);
      }
    }

    void replaceKeywords(exprNode &expr,
                         const keyword_t &from,
                         keyword_t &to) {
      const int kType = from.type();
      if (kType & keywordType::type) {
        const type_t &fromValue = ((const typeKeyword&) from).type_;
        type_t &toValue = ((typeKeyword&) to).type_;
        replaceTypes(expr, fromValue, toValue);
      }
      else if (kType & keywordType::variable) {
        const variable_t &fromValue = ((const variableKeyword&) from).variable;
        variable_t &toValue = ((variableKeyword&) to).variable;
        replaceVariables(expr, fromValue, toValue);
      }
      else if (kType & keywordType::function) {
        const function_t &fromValue = ((const functionKeyword&) from).function;
        function_t &toValue = ((functionKeyword&) to).function;
        replaceFunctions(expr, fromValue, toValue);
      }
    }
  }
}
