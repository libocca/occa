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
#ifndef OCCA_LANG_BUILTINS_TYPES_HEADER
#define OCCA_LANG_BUILTINS_TYPES_HEADER

#include <occa/lang/type.hpp>

namespace occa {
  namespace lang {
    extern const qualifier_t const_;
    extern const qualifier_t constexpr_;
    extern const qualifier_t restrict_;
    extern const qualifier_t friend_;
    extern const qualifier_t typedef_;
    extern const qualifier_t signed_;
    extern const qualifier_t unsigned_;
    extern const qualifier_t volatile_;
    extern const qualifier_t long_;
    extern const qualifier_t longlong_;

    extern const qualifier_t extern_;
    extern const qualifier_t externC;
    extern const qualifier_t externCpp;
    extern const qualifier_t mutable_;
    extern const qualifier_t register_;
    extern const qualifier_t static_;
    extern const qualifier_t thread_local_;

    extern const qualifier_t explicit_;
    extern const qualifier_t inline_;
    extern const qualifier_t virtual_;

    extern const qualifier_t class_;
    extern const qualifier_t struct_;
    extern const qualifier_t enum_;
    extern const qualifier_t union_;

    extern const primitive_t bool_;
    extern const primitive_t char_;
    extern const primitive_t char16_t_;
    extern const primitive_t char32_t_;
    extern const primitive_t wchar_t_;
    extern const primitive_t short_;
    extern const primitive_t int_;
    extern const primitive_t float_;
    extern const primitive_t double_;
    extern const primitive_t void_;
    extern const primitive_t auto_;
  }
}
#endif
