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

namespace occa {
  namespace lang {
    scope_t::scope_t(context &ctx_) :
      ctx(ctx_) {}

    void scope_t::add(typedefType &value) {
      add(value, keywordType::typedef_);
    }

    void scope_t::add(classType &value) {
      add(value, keywordType::class_);
    }

    void scope_t::add(functionType &value) {
      add(value, keywordType::function_);
    }

    void scope_t::add(attribute_t &value) {
      add(value, keywordType::attribute);
    }

    void scope_t::add(specifier &value, const int ktype) {
      // Ctx checks for duplicates
      ctx.add(value, ktype);
      trie.add(value.uniqueName(),
               keyword_t(ktype, &value));
    }

    keyword_t scope_t::get(const std::string &name) {
      return trie.get(name).value();
    }
  }
}
