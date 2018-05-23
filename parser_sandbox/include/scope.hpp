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
#ifndef OCCA_LANG_SCOPE_HEADER
#define OCCA_LANG_SCOPE_HEADER

#include <map>

#include "occa/defines.hpp"

namespace occa {
  namespace lang {
    class type_t;
    class function_t;
    class variable_t;
    class scopeKeyword_t;

    typedef std::map<std::string, scopeKeyword_t> scopeKeywordMap_t;
    typedef scopeKeywordMap_t::iterator           scopeKeywordMapIterator;

    namespace scopeKeywordType {
      extern const int none;
      extern const int type;
      extern const int function;
      extern const int variable;
    }

    class scopeKeyword_t {
    private:
      int sktype;
      void *ptr;

    public:
      scopeKeyword_t();
      scopeKeyword_t(type_t &t);
      scopeKeyword_t(function_t &func);
      scopeKeyword_t(variable_t &var);

      scopeKeyword_t clone();

      bool exists() const;
      bool isType() const;
      bool isFunction() const;
      bool isVariable() const;

      type_t& type();
      function_t& function();
      variable_t& variable();

      friend class scope_t;
    };

    class scope_t {
    public:
      scopeKeywordMap_t keywordMap;

      scope_t();
      ~scope_t();

      void clear();

      int size();

      bool has(const std::string &name);
      scopeKeyword_t get(const std::string &name);

      bool add(const type_t &type);
      bool add(const function_t &func);
      bool add(const variable_t &var);
    };
  }
}

#endif
