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
#ifndef OCCA_LANG_BUILTINS_ATTRIBUTES_HEADER
#define OCCA_LANG_BUILTINS_ATTRIBUTES_HEADER

#include "attribute.hpp"

namespace occa {
  namespace lang {
    //---[ @dim ]-----------------------
    class dim : public attribute_t {
    public:
      dim();
      dim(identifierToken &source_);
      virtual ~dim();

      virtual std::string name() const;

      virtual attribute_t* create(parser_t &parser,
                                  identifierToken &source_,
                                  const tokenRangeVector &argRanges);

      virtual attribute_t* clone();

      virtual bool isVariableAttribute() const;

      virtual bool onVariableLoad(parser_t &parser,
                                  variable_t &var);
    };
    //==================================

    //---[ @dimOrder ]------------------
    class dimOrder : public attribute_t {
    public:
      dimOrder();
      dimOrder(identifierToken &source_);
      virtual ~dimOrder();

      virtual std::string name() const;

      virtual attribute_t* create(parser_t &parser,
                                  identifierToken &source_,
                                  const tokenRangeVector &argRanges);

      virtual attribute_t* clone();

      virtual bool isVariableAttribute() const;

      virtual bool onVariableLoad(parser_t &parser,
                                  variable_t &var);
    };
    //==================================

    //---[ @tile ]----------------------
    class tile : public attribute_t {
    public:
      tile();
      tile(identifierToken &source_);
      virtual ~tile();

      virtual std::string name() const;

      virtual attribute_t* create(parser_t &parser,
                                  identifierToken &source_,
                                  const tokenRangeVector &argRanges);

      virtual attribute_t* clone();

      virtual bool isStatementAttribute(const int stype) const;

      virtual bool onStatementLoad(parser_t &parser,
                                   statement_t &smnt);
    };
    //==================================

    //---[ @safeTile ]------------------
    class safeTile : public attribute_t {
    public:
      safeTile();
      safeTile(identifierToken &source_);
      virtual ~safeTile();

      virtual std::string name() const;

      virtual attribute_t* create(parser_t &parser,
                                  identifierToken &source_,
                                  const tokenRangeVector &argRanges);

      virtual attribute_t* clone();

      virtual bool isStatementAttribute(const int stype) const;

      virtual bool onStatementLoad(parser_t &parser,
                                   statement_t &smnt);
    };
    //==================================
  }
}
#endif
