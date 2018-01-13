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
#ifndef OCCA_PARSER_TYPEBUILTINS_HEADER2
#define OCCA_PARSER_TYPEBUILTINS_HEADER2

#include "type.hpp"

namespace occa {
  namespace lang {
    extern const qualifier const_;
    extern const qualifier constexpr_;
    extern const qualifier friend_;
    extern const qualifier typedef_;
    extern const qualifier signed_;
    extern const qualifier unsigned_;
    extern const qualifier volatile_;

    extern const qualifier extern_;
    extern const qualifier mutable_;
    extern const qualifier register_;
    extern const qualifier static_;
    extern const qualifier thread_local_;

    extern const qualifier explicit_;
    extern const qualifier inline_;
    extern const qualifier virtual_;

    extern const qualifier class_;
    extern const qualifier struct_;
    extern const qualifier enum_;
    extern const qualifier union_;

    extern const primitiveType bool_;
    extern const primitiveType char_;
    extern const primitiveType char16_t_;
    extern const primitiveType char32_t_;
    extern const primitiveType wchar_t_;
    extern const primitiveType short_;
    extern const primitiveType int_;
    extern const primitiveType long_;
    extern const primitiveType float_;
    extern const primitiveType double_;
    extern const primitiveType void_;
    extern const primitiveType auto_;
  }
}
#endif
