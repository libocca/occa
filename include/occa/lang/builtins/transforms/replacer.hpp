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

#ifndef OCCA_LANG_BUILTINS_TRANSFORMS_REPLACER_HEADER
#define OCCA_LANG_BUILTINS_TRANSFORMS_REPLACER_HEADER

#include <occa/lang/exprTransform.hpp>
#include <occa/lang/builtins/transforms/finders.hpp>

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
