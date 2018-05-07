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
#ifndef OCCA_LANG_VARIABLE_HEADER
#define OCCA_LANG_VARIABLE_HEADER

#include "type.hpp"

namespace occa {
  namespace lang {
    class attribute_t;

    //---[ Variable ]-------------------
    class variable_t {
    public:
      vartype_t vartype;
      identifierToken *source;
      attributePtrVector attributes;

      variable_t();

      variable_t(const vartype_t &vartype_,
                 identifierToken *source_ = NULL);

      variable_t(const variable_t &other);
      variable_t& operator = (const variable_t &other);

      ~variable_t();

      bool isNamed() const;
      const std::string& name() const;

      bool operator == (const variable_t &other) const;

      void printDeclaration(printer &pout) const;
      void printExtraDeclaration(printer &pout) const;
    };

    printer& operator << (printer &pout,
                          const variable_t &var);
    //==================================

    //---[ Variable Declaration ]-------
    class variableDeclaration {
    public:
      variable_t var;
      exprNode *value;

      variableDeclaration();

      variableDeclaration(const variable_t &var_);

      variableDeclaration(const variable_t &var_,
                          exprNode &value_);

      variableDeclaration(const variableDeclaration &other);

      ~variableDeclaration();

      bool hasValue() const;

      void print(printer &pout) const;
      void printAsExtra(printer &pout) const;
    };
    //==================================
  }
}

#endif
