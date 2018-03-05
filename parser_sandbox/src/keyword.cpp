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
#include "keyword.hpp"
#include "typeBuiltins.hpp"

namespace occa {
  namespace lang {
    keyword_t::keyword_t() :
      ptr(NULL),
      name(),
      kType(keywordType::none) {}

    keyword_t::keyword_t(const qualifier &value) :
      ptr(const_cast<void*>((void*) &value)),
      name(value.name),
      kType(keywordType::qualifier) {}

    keyword_t::keyword_t(const primitiveType &value) :
      ptr(const_cast<void*>((void*) &value)),
      name(value.name),
      kType(keywordType::primitiveType) {}

    keyword_t::keyword_t(type_t &type) :
      ptr(&type),
      name(type.name),
      kType(keywordType::type) {}

    keyword_t::keyword_t(functionType &type) :
      ptr(&type),
      name(type.name),
      kType(keywordType::function) {}

    keyword_t::keyword_t(void *ptr_,
                         const std::string &name_,
                         const int kType_) :
      ptr(ptr_),
      name(name_),
      kType(kType_) {}

    void keyword_t::free() {
      // Builtins
      if (kType & (keywordType::type |
                   keywordType::function)) {
        // Custom types
        if (kType & keywordType::type) {
          delete (type_t*) ptr;
        }
        else if (kType & keywordType::function) {
          delete (functionType*) ptr;
        }
      }
      ptr   = NULL;
      name  = "";
      kType = keywordType::none;
    }

    void getKeywords(keywordTrie &keywords) {
      addKeyword(keywords, const_);
      addKeyword(keywords, constexpr_);
      addKeyword(keywords, friend_);
      addKeyword(keywords, typedef_);
      addKeyword(keywords, signed_);
      addKeyword(keywords, unsigned_);
      addKeyword(keywords, volatile_);

      addKeyword(keywords, extern_);
      addKeyword(keywords, mutable_);
      addKeyword(keywords, register_);
      addKeyword(keywords, static_);
      addKeyword(keywords, thread_local_);

      addKeyword(keywords, explicit_);
      addKeyword(keywords, inline_);
      addKeyword(keywords, virtual_);

      addKeyword(keywords, class_);
      addKeyword(keywords, struct_);
      addKeyword(keywords, enum_);
      addKeyword(keywords, union_);

      addKeyword(keywords, bool_);
      addKeyword(keywords, char_);
      addKeyword(keywords, char16_t_);
      addKeyword(keywords, char32_t_);
      addKeyword(keywords, wchar_t_);
      addKeyword(keywords, short_);
      addKeyword(keywords, int_);
      addKeyword(keywords, long_);
      addKeyword(keywords, float_);
      addKeyword(keywords, double_);
      addKeyword(keywords, void_);
      addKeyword(keywords, auto_);
    }

    void freeKeywords(keywordTrie &keywords) {
      keywords.freeze();
      const int count = keywords.size();
      for (int i = 0; i < count; ++i) {
        keywords.values[i].free();
      }
      keywords.clear();
    }
  }
}
