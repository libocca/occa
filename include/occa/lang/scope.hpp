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

#include <occa/lang/keyword.hpp>

namespace occa {
  namespace lang {
    // Note: scope_t doesn't clone keywords
    class scope_t {
    public:
      keywordMap keywords;

      scope_t();
      ~scope_t();

      void clear();

      scope_t clone() const;

      void swap(scope_t &other);

      int size();

      bool has(const std::string &name);
      keyword_t& get(const std::string &name);

      bool add(type_t &type,
               const bool force = false);
      bool add(function_t &func,
               const bool force = false);
      bool add(variable_t &var,
               const bool force = false);
    private:
      template <class keywordType_,
                class valueType>
      bool add(valueType &value,
               const bool force) {
        // TODO: Use unique name
        const std::string &name = value.name();
        if (!name.size()) {
          return true;
        }
        keywordMapIterator it = keywords.find(name);
        if (it == keywords.end()) {
          keywords[name] = new keywordType_(value);
          return true;
        }
        keyword_t *&keyword = it->second;
        if (force) {
          keyword->deleteSource();
          delete keyword;
          keyword = new keywordType_(value);
          return true;
        }
        value.printError("[" + name + "] is already defined");
        keyword->printError("[" + name + "] was first defined here");
        return false;
      }

    public:
      void remove(const std::string &name,
                  const bool deleteSource = true);

      void moveTo(scope_t &scope);

      void debugPrint() const;
    };
  }
}

#endif
