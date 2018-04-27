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
#include <sstream>

#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"

#include "type.hpp"
#include "statement.hpp"
#include "variable.hpp"
#include "expression.hpp"
#include "builtins/types.hpp"

namespace occa {
  namespace lang {
    namespace qualifierType {
      const int none          = 0;

      const int auto_         = (1L << 0);
      const int const_        = (1L << 1);
      const int constexpr_    = (1L << 2);
      const int restrict_     = (1L << 3);
      const int signed_       = (1L << 4);
      const int unsigned_     = (1L << 5);
      const int volatile_     = (1L << 6);
      const int long_         = (1L << 7);
      const int longlong_     = (1L << 8);
      const int register_     = (1L << 9);
      const int typeInfo      = (const_     |
                                 constexpr_ |
                                 signed_    |
                                 unsigned_  |
                                 volatile_  |
                                 long_      |
                                 longlong_  |
                                 register_);

      const int forPointers   = (const_    |
                                 restrict_ |
                                 volatile_);

      const int extern_       = (1L << 10);
      const int static_       = (1L << 11);
      const int thread_local_ = (1L << 12);
      const int globalScope   = (extern_ |
                                 static_ |
                                 thread_local_);

      const int friend_       = (1L << 13);
      const int mutable_      = (1L << 14);
      const int classInfo     = (friend_ |
                                 mutable_);

      const int inline_       = (1L << 15);
      const int virtual_      = (1L << 16);
      const int explicit_     = (1L << 17);
      const int functionInfo  = (typeInfo |
                                 inline_  |
                                 virtual_ |
                                 explicit_);

      const int builtin_      = (1L << 18);
      const int typedef_      = (1L << 19);
      const int class_        = (1L << 20);
      const int enum_         = (1L << 21);
      const int struct_       = (1L << 22);
      const int union_        = (1L << 23);
      const int newType       = (typedef_ |
                                 class_   |
                                 enum_    |
                                 struct_  |
                                 union_);
    }

    //---[ Qualifier ]------------------
    qualifier_t::qualifier_t(const std::string &name_,
                             const int qtype_) :
      name(name_),
      qtype(qtype_) {}

    qualifier_t::~qualifier_t() {}

    int qualifier_t::type() const {
      return qtype;
    }

    printer& operator << (printer &pout,
                          const qualifier_t &qualifier) {
      pout << qualifier.name;
      return pout;
    }
    //==================================

    //---[ Qualifiers ]-----------------
    qualifierWithSource::qualifierWithSource(const qualifier_t &qualifier_) :
      origin(),
      qualifier(&qualifier_) {}

    qualifierWithSource::qualifierWithSource(const fileOrigin &origin_,
                                             const qualifier_t &qualifier_) :
      origin(origin_),
      qualifier(&qualifier_) {}

    qualifiers_t::qualifiers_t() {}

    qualifiers_t::~qualifiers_t() {}

    void qualifiers_t::clear() {
      qualifiers.clear();
    }

    const qualifier_t* qualifiers_t::operator [] (const int index) {
      if ((index < 0) ||
          (index >= (int) qualifiers.size())) {
        return NULL;
      }
      return qualifiers[index].qualifier;
    }

    int qualifiers_t::indexOf(const qualifier_t &qualifier) const {
      const int count = (int) qualifiers.size();
      if (count) {
        for (int i = 0; i < count; ++i) {
          if (qualifiers[i].qualifier == &qualifier) {
            return i;
          }
        }
      }
      return -1;
    }

    bool qualifiers_t::has(const qualifier_t &qualifier) const {
      return (indexOf(qualifier) >= 0);
    }

    bool qualifiers_t::operator == (const qualifiers_t &other) const {
      const int count      = (int) qualifiers.size();
      const int otherCount = (int) other.qualifiers.size();
      if (count != otherCount) {
        return false;
      }
      for (int i = 0; i < count; ++i) {
        if (!other.has(*(qualifiers[i].qualifier))) {
          return false;
        }
      }
      return true;
    }

    bool qualifiers_t::operator != (const qualifiers_t &other) const {
      return !((*this) == other);
    }

    qualifiers_t& qualifiers_t::operator += (const qualifier_t &qualifier) {
      if (!has(qualifier)) {
        qualifiers.push_back(qualifier);
      }
      return *this;
    }

    qualifiers_t& qualifiers_t::operator -= (const qualifier_t &qualifier) {
      const int idx = indexOf(qualifier);
      if (idx >= 0) {
        qualifiers.erase(qualifiers.begin() + idx);
      }
      return *this;
    }

    qualifiers_t& qualifiers_t::operator += (const qualifiers_t &other) {
      const int count = (int) other.qualifiers.size();
      for (int i = 0; i < count; ++i) {
        this->add(other.qualifiers[i]);
      }
      return *this;
    }

    qualifiers_t& qualifiers_t::add(const fileOrigin &origin,
                                    const qualifier_t &qualifier) {
      if (!has(qualifier)) {
        qualifiers.push_back(qualifierWithSource(origin, qualifier));
      }
      return *this;
    }

    qualifiers_t& qualifiers_t::add(const qualifierWithSource &qualifier) {
      if (!has(*(qualifier.qualifier))) {
        qualifiers.push_back(qualifier);
      }
      return *this;
    }

    printer& operator << (printer &pout,
                          const qualifiers_t &qualifiers) {
      const qualifierVector_t &quals = qualifiers.qualifiers;

      const int count = (int) quals.size();
      if (!count) {
        return pout;
      }
      pout << *(quals[0].qualifier);
      for (int i = 1; i < count; ++i) {
        pout << ' ' << *(quals[i].qualifier);
      }
      return pout;
    }
    //==================================
  }
}
