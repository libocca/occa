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
#include <occa/lang/scope.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/type.hpp>

namespace occa {
  namespace lang {
    scope_t::scope_t() {}

    scope_t::~scope_t() {
      clear();
    }

    void scope_t::clear() {
      freeKeywords(keywords, true);
    }

    scope_t scope_t::clone() const {
      scope_t other;
      keywordMap::const_iterator it = keywords.begin();
      while (it != keywords.end()) {
        other.keywords[it->first] = it->second->clone();
        ++it;
      }
      return other;
    }

    void scope_t::swap(scope_t &other) {
      keywords.swap(other.keywords);
    }

    int scope_t::size() {
      return (int) keywords.size();
    }

    bool scope_t::has(const std::string &name) {
      return (keywords.find(name) != keywords.end());
    }

    keyword_t& scope_t::get(const std::string &name) {
      static keyword_t noKeyword;
      keywordMapIterator it = keywords.find(name);
      if (it != keywords.end()) {
        return *it->second;
      }
      return noKeyword;
    }

    bool scope_t::add(type_t &type,
                      const bool force) {
      return add<typeKeyword>(type, force);
    }

    bool scope_t::add(function_t &func,
                      const bool force) {
      return add<functionKeyword>(func, force);
    }

    bool scope_t::add(variable_t &var,
                      const bool force) {
      return add<variableKeyword>(var, force);
    }

    void scope_t::remove(const std::string &name,
                         const bool deleteSource) {
      keywordMapIterator it = keywords.find(name);
      if (it != keywords.end()) {
        keyword_t &keyword = *(it->second);
        if (deleteSource) {
          keyword.deleteSource();
        }
        delete &keyword;
        keywords.erase(it);
      }
    }

    void scope_t::moveTo(scope_t &scope) {
      scope.keywords.insert(keywords.begin(),
                            keywords.end());
      keywords.clear();
    }

    void scope_t::debugPrint() const {
      keywordMap::const_iterator it = keywords.begin();
      while (it != keywords.end()) {
        std::cout << '['
                  << stringifySetBits(it->second->type())
                  << "] "
                  << it->first << '\n';
        ++it;
      }
    }
  }
}
