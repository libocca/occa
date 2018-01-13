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
#ifndef OCCA_PARSER_KEYWORD_HEADER2
#define OCCA_PARSER_KEYWORD_HEADER2

#include "occa/defines.hpp"
#include "occa/types.hpp"
#include "occa/tools/sys.hpp"

#include "type.hpp"

namespace occa {
  namespace lang {
    class keywordType {
    public:
      static const int none      = 0;
      static const int qualifier = (1 << 0);
      static const int primitive = (1 << 1);
      static const int typedef_  = (1 << 2);
      static const int class_    = (1 << 3);
      static const int function_ = (1 << 4);
      static const int attribute = (1 << 5);
    };

    class keyword_t {
    public:
      int ktype;
      specifier *ptr;

      keyword_t();
      keyword_t(const int ktype_, specifier *ptr_);

      inline bool isQualifier() {
        return (ktype == keywordType::qualifier);
      }

      inline bool isPrimitive() {
        return (ktype == keywordType::primitive);
      }

      inline bool isTypedef_() {
        return (ktype == keywordType::typedef_);
      }

      inline bool isClass_() {
        return (ktype == keywordType::class_);
      }

      inline bool isFunction_() {
        return (ktype == keywordType::function_);
      }

      inline bool isAttribute() {
        return (ktype == keywordType::attribute);
      }

      inline class qualifier& qualifier() {
        return *((class qualifier*) ptr);
      }

      inline primitiveType& primitive() {
        return *((primitiveType*) ptr);
      }

      inline typedefType& typedef_() {
        return *dynamic_cast<typedefType*>(ptr);
      }

      inline classType& class_() {
        return *dynamic_cast<classType*>(ptr);
      }

      inline functionType& function_() {
        return *dynamic_cast<functionType*>(ptr);
      }

      inline class attribute& attribute() {
        return *((class attribute*) ptr);
      }
    };
  }
}

#endif
