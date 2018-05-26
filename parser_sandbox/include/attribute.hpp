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
#ifndef OCCA_LANG_ATTRIBUTE_HEADER
#define OCCA_LANG_ATTRIBUTE_HEADER

#include <iostream>
#include <vector>
#include <map>

#include "tokenContext.hpp"

namespace occa {
  namespace lang {
    class parser_t;
    class identifierToken;
    class attribute_t;
    class attributeToken_t;
    class vartype_t;
    class variable_t;
    class function_t;
    class statement_t;
    class expressionStatement;

    typedef std::vector<exprNode*>                  exprNodeVector;
    typedef std::map<std::string, exprNode*>        exprNodeMap;
    typedef std::map<std::string, attributeToken_t> attributeTokenMap;

    //---[ Attribute Type ]-------------
    class attribute_t {
    public:
      virtual ~attribute_t();

      virtual std::string name() const = 0;

      virtual bool forVariable() const;
      virtual bool forFunction() const;
      virtual bool forStatement(const int sType) const = 0;

      virtual bool isValid(const attributeToken_t &attr) const = 0;
    };
    //==================================

    //---[ Attribute ]------------------
    class attributeToken_t {
    public:
      const attribute_t *attrType;
      identifierToken *source;
      exprNodeVector args;
      exprNodeMap kwargs;

      attributeToken_t();
      attributeToken_t(const attribute_t &attrType_,
                       identifierToken &source_);
      attributeToken_t(const attributeToken_t &other);
      attributeToken_t& operator = (const attributeToken_t &other);
      virtual ~attributeToken_t();

      const std::string& name() const;

      bool forVariable() const;
      bool forFunction() const;
      bool forStatement(const int sType) const;

      exprNode* operator [] (const int index);
      exprNode* operator [] (const std::string &arg);

      void printWarning(const std::string &message) const;
      void printError(const std::string &message) const;
    };
    //==================================
  }
}

#endif
