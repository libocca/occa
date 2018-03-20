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
#include "typeBuiltins.hpp"

namespace occa {
  namespace lang {
    const qualifier_t const_        ("const"       , qualifierType::const_);
    const qualifier_t constexpr_    ("constexpr"   , qualifierType::constexpr_);
    const qualifier_t friend_       ("friend"      , qualifierType::friend_);
    const qualifier_t typedef_      ("typedef"     , qualifierType::typedef_);
    const qualifier_t signed_       ("signed"      , qualifierType::signed_);
    const qualifier_t unsigned_     ("unsigned"    , qualifierType::unsigned_);
    const qualifier_t volatile_     ("volatile"    , qualifierType::volatile_);
    const qualifier_t long_         ("long"        , qualifierType::long_);
    const qualifier_t longlong_     ("long long"   , qualifierType::longlong_);

    const qualifier_t extern_       ("extern"      , qualifierType::extern_);
    const qualifier_t mutable_      ("mutable"     , qualifierType::mutable_);
    const qualifier_t register_     ("register"    , qualifierType::register_);
    const qualifier_t static_       ("static"      , qualifierType::static_);
    const qualifier_t thread_local_ ("thread_local", qualifierType::thread_local_);

    const qualifier_t explicit_     ("explicit"    , qualifierType::explicit_);
    const qualifier_t inline_       ("inline"      , qualifierType::inline_);
    const qualifier_t virtual_      ("virtual"     , qualifierType::virtual_);

    const qualifier_t class_        ("class"       , qualifierType::class_);
    const qualifier_t enum_         ("enum"        , qualifierType::enum_);
    const qualifier_t struct_       ("struct"      , qualifierType::struct_);
    const qualifier_t union_        ("union"       , qualifierType::union_);

    const primitive_t bool_         ("bool");
    const primitive_t char_         ("char");
    const primitive_t char16_t_     ("char16_t");
    const primitive_t char32_t_     ("char32_t");
    const primitive_t wchar_t_      ("wchar_t");
    const primitive_t short_        ("short");
    const primitive_t int_          ("int");
    const primitive_t float_        ("float");
    const primitive_t double_       ("double");
    const primitive_t void_         ("void");
    const primitive_t auto_         ("auto");
  }
}
