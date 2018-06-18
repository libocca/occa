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

#include <occa/lang/primitive.hpp>
#include <occa/lang/file.hpp>
#include <occa/lang/printer.hpp>

namespace occa {
  namespace lang {
    namespace qualifierType {
      extern const udim_t none;

      extern const udim_t auto_;
      extern const udim_t const_;
      extern const udim_t constexpr_;
      extern const udim_t signed_;
      extern const udim_t unsigned_;
      extern const udim_t volatile_;
      extern const udim_t register_;
      extern const udim_t long_;
      extern const udim_t longlong_;
      extern const udim_t typeInfo;

      extern const udim_t forPointers_;
      extern const udim_t forPointers;

      extern const udim_t extern_;
      extern const udim_t externC;
      extern const udim_t externCpp;
      extern const udim_t static_;
      extern const udim_t thread_local_;

      extern const udim_t globalScope_;
      extern const udim_t globalScope;

      extern const udim_t friend_;
      extern const udim_t mutable_;

      extern const udim_t classInfo_;
      extern const udim_t classInfo;

      extern const udim_t inline_;
      extern const udim_t virtual_;
      extern const udim_t explicit_;

      extern const udim_t functionInfo_;
      extern const udim_t functionInfo;

      extern const udim_t builtin_;
      extern const udim_t typedef_;
      extern const udim_t class_;
      extern const udim_t enum_;
      extern const udim_t struct_;
      extern const udim_t union_;

      extern const udim_t newType_;
      extern const udim_t newType;

      extern const udim_t custom;
    }

    //---[ Qualifier ]------------------
    class qualifier_t {
    public:
      std::string name;
      const udim_t qtype;

      qualifier_t(const std::string &name_,
                  const udim_t qtype_);
      ~qualifier_t();

      udim_t type() const;
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

      void printWarning(const std::string &message) const;
      void printError(const std::string &message) const;
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

      const qualifier_t* operator [] (const int index) const;

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

      qualifiers_t& addFirst(const fileOrigin &origin,
                             const qualifier_t &qualifier);

      qualifiers_t& addFirst(const qualifierWithSource &qualifier);
    };

    printer& operator << (printer &pout,
                          const qualifiers_t &qualifiers);
    //==================================
  }
}
#endif
