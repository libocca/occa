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
#include "scope.hpp"
#include "variable.hpp"
#include "type.hpp"

namespace occa {
  namespace lang {
    namespace scopeKeywordType {
      const int none     = 0;
      const int type     = 1;
      const int function = 2;
      const int variable = 3;
    }

    scopeKeyword_t::scopeKeyword_t() :
      sktype(scopeKeywordType::none),
      ptr(NULL) {}

    scopeKeyword_t::scopeKeyword_t(type_t &t) :
      sktype(scopeKeywordType::type),
      ptr(&t) {}

    scopeKeyword_t::scopeKeyword_t(function_t &func) :
      sktype(scopeKeywordType::function),
      ptr(&func) {}

    scopeKeyword_t::scopeKeyword_t(variable_t &var) :
      sktype(scopeKeywordType::variable),
      ptr(&var) {}

    bool scopeKeyword_t::exists() const {
      return (sktype != scopeKeywordType::none);
    }

    bool scopeKeyword_t::isType() const {
      return (sktype == scopeKeywordType::type);
    }

    bool scopeKeyword_t::isFunction() const {
      return (sktype == scopeKeywordType::function);
    }

    bool scopeKeyword_t::isVariable() const {
      return (sktype == scopeKeywordType::variable);
    }

    type_t& scopeKeyword_t::type() {
      return *((type_t*) ptr);
    }

    function_t& scopeKeyword_t::function() {
      return *((function_t*) ptr);
    }

    variable_t& scopeKeyword_t::variable() {
      return *((variable_t*) ptr);
    }

    scope_t::scope_t() {}

    scope_t::~scope_t() {
      clear();
    }

    void scope_t::clear() {
      scopeKeywordMapIterator it = keywordMap.begin();
      while (it != keywordMap.end()) {
        scopeKeyword_t &keyword = it->second;
        switch (keyword.sktype) {
        case scopeKeywordType::type: {
          delete (type_t*) keyword.ptr;
          break;
        }
        case scopeKeywordType::function:
          delete (function_t*) keyword.ptr;
          break;
        case scopeKeywordType::variable:
          delete (variable_t*) keyword.ptr;
          break;
        }
        ++it;
      }
    }

    int scope_t::size() {
      return (int) keywordMap.size();
    }

    bool scope_t::has(const std::string &name) {
      return (keywordMap.find(name) != keywordMap.end());
    }

    scopeKeyword_t scope_t::get(const std::string &name) {
      scopeKeywordMapIterator it = keywordMap.find(name);
      if (it != keywordMap.end()) {
        return it->second;
      }
      return scopeKeyword_t();
    }

    void scope_t::add(const type_t &type) {
      const std::string &name = type.name();
      if (name.size()) {
        keywordMap[name] = scopeKeyword_t(type.clone());
      }
    }

    void scope_t::add(const function_t &func) {
      const std::string &name = func.name();
      if (name.size()) {
        keywordMap[name] = scopeKeyword_t(func.clone().to<function_t>());
      }
    }

    void scope_t::add(const variable_t &var) {
      const std::string &name = var.name();
      if (name.size()) {
        keywordMap[name] = scopeKeyword_t(*(new variable_t(var)));
      }
    }
  }
}
