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
#ifndef OCCA_LANG_QUALIFIER_HEADER
#define OCCA_LANG_QUALIFIER_HEADER

#include <vector>

#include "occa/parser/primitive.hpp"
#include "file.hpp"
#include "printer.hpp"

namespace occa {
  namespace lang {
    namespace qualifierType {
      extern const int none;

      extern const int auto_;
      extern const int const_;
      extern const int constexpr_;
      extern const int restrict_;
      extern const int signed_;
      extern const int unsigned_;
      extern const int volatile_;
      extern const int register_;
      extern const int long_;
      extern const int longlong_;
      extern const int typeInfo;

      extern const int forPointers;

      extern const int extern_;
      extern const int static_;
      extern const int thread_local_;
      extern const int globalScope;

      extern const int friend_;
      extern const int mutable_;
      extern const int classInfo;

      extern const int inline_;
      extern const int virtual_;
      extern const int explicit_;
      extern const int functionInfo;

      extern const int builtin_;
      extern const int typedef_;
      extern const int class_;
      extern const int enum_;
      extern const int struct_;
      extern const int union_;
      extern const int newType;
    }

    //---[ Qualifier ]------------------
    class qualifier_t {
    public:
      std::string name;
      const int qtype;

      qualifier_t(const std::string &name_,
                  const int qtype_);
      ~qualifier_t();

      int type() const;
    };

    printer& operator << (printer &pout,
                          const qualifier_t &qualifier);
    //==================================

    //---[ Qualifiers ]-----------------
    class qualifierWithSource {
    public:
      fileOrigin origin;
      const qualifier_t *qualifier;

      qualifierWithSource(const qualifier_t &qualifier_);

      qualifierWithSource(const fileOrigin &origin_,
                          const qualifier_t &qualifier_);
    };

    typedef std::vector<qualifierWithSource> qualifierVector_t;

    class qualifiers_t {
    public:
      qualifierVector_t qualifiers;

      qualifiers_t();
      ~qualifiers_t();

      void clear();

      inline int size() const {
        return (int) qualifiers.size();
      }

      const qualifier_t* operator [] (const int index);

      int indexOf(const qualifier_t &qualifier) const;
      bool has(const qualifier_t &qualifier) const;

      bool operator == (const qualifiers_t &other) const;
      bool operator != (const qualifiers_t &other) const;

      qualifiers_t& operator += (const qualifier_t &qualifier);
      qualifiers_t& operator -= (const qualifier_t &qualifier);
      qualifiers_t& operator += (const qualifiers_t &others);

      qualifiers_t& add(const fileOrigin &origin,
                        const qualifier_t &qualifier);

      qualifiers_t& add(const qualifierWithSource &qualifier);
    };

    printer& operator << (printer &pout,
                          const qualifiers_t &qualifiers);
    //==================================
  }
}
#endif
