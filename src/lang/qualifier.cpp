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

#include <occa/defines.hpp>
#include <occa/tools/sys.hpp>

#include <occa/lang/type.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/expression.hpp>
#include <occa/lang/builtins/types.hpp>

namespace occa {
  namespace lang {
    namespace qualifierType {
      const udim_t none          = (1L << 0);

      const udim_t auto_         = (1L << 1);
      const udim_t const_        = (1L << 2);
      const udim_t constexpr_    = (1L << 3);
      const udim_t signed_       = (1L << 4);
      const udim_t unsigned_     = (1L << 5);
      const udim_t volatile_     = (1L << 6);
      const udim_t long_         = (1L << 7);
      const udim_t longlong_     = (1L << 8);
      const udim_t register_     = (1L << 9);

      const udim_t typeInfo_     = (1L << 10);
      const udim_t typeInfo      = (const_     |
                                    constexpr_ |
                                    signed_    |
                                    unsigned_  |
                                    volatile_  |
                                    long_      |
                                    longlong_  |
                                    register_  |
                                    typeInfo_);

      const udim_t forPointers_  = (1L << 11);
      const udim_t forPointers   = (const_    |
                                    volatile_ |
                                    forPointers_);

      const udim_t extern_       = (1L << 12);
      const udim_t externC       = (1L << 13);
      const udim_t externCpp     = (1L << 14);
      const udim_t static_       = (1L << 15);
      const udim_t thread_local_ = (1L << 16);

      const udim_t globalScope_  = (1L << 17);
      const udim_t globalScope   = (extern_       |
                                    externC       |
                                    externCpp     |
                                    static_       |
                                    thread_local_ |
                                    globalScope_);

      const udim_t friend_       = (1L << 18);
      const udim_t mutable_      = (1L << 19);

      const udim_t classInfo_    = (1L << 20);
      const udim_t classInfo     = (friend_  |
                                    mutable_ |
                                    classInfo_);

      const udim_t inline_       = (1L << 21);
      const udim_t virtual_      = (1L << 22);
      const udim_t explicit_     = (1L << 23);

      const udim_t functionInfo_ = (1L << 24);
      const udim_t functionInfo  = (typeInfo  |
                                    inline_   |
                                    virtual_  |
                                    explicit_ |
                                    functionInfo_);

      const udim_t builtin_      = (1L << 25);
      const udim_t typedef_      = (1L << 26);
      const udim_t class_        = (1L << 27);
      const udim_t enum_         = (1L << 28);
      const udim_t struct_       = (1L << 29);
      const udim_t union_        = (1L << 30);

      const udim_t newType_      = (1L << 31);
      const udim_t newType       = (typedef_ |
                                    class_   |
                                    enum_    |
                                    struct_  |
                                    union_   |
                                    newType_);

      const udim_t custom        = (1L << 32);
    }

    //---[ Qualifier ]------------------
    qualifier_t::qualifier_t(const std::string &name_,
                             const udim_t qtype_) :
      name(name_),
      qtype(qtype_) {}

    qualifier_t::~qualifier_t() {}

    udim_t qualifier_t::type() const {
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

    void qualifierWithSource::printWarning(const std::string &message) const {
      origin.printWarning(message);
    }
    void qualifierWithSource::printError(const std::string &message) const {
      origin.printError(message);
    }

    qualifiers_t::qualifiers_t() {}

    qualifiers_t::~qualifiers_t() {}

    void qualifiers_t::clear() {
      qualifiers.clear();
    }

    const qualifier_t* qualifiers_t::operator [] (const int index) const {
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
        qualifiers.push_back(
          qualifierWithSource(origin, qualifier)
        );
      }
      return *this;
    }

    qualifiers_t& qualifiers_t::add(const qualifierWithSource &qualifier) {
      if (!has(*(qualifier.qualifier))) {
        qualifiers.push_back(qualifier);
      }
      return *this;
    }

    qualifiers_t& qualifiers_t::addFirst(const fileOrigin &origin,
                                         const qualifier_t &qualifier) {
      return addFirst(
        qualifierWithSource(origin, qualifier)
      );
    }

    qualifiers_t& qualifiers_t::addFirst(const qualifierWithSource &qualifier) {
      if (has(*(qualifier.qualifier))) {
        return *this;
      }
      const int count = (int) qualifiers.size();
      if (!count) {
        qualifiers.push_back(qualifier);
        return *this;
      }

      qualifiers.push_back(qualifiers[count - 1]);
      for (int i = 0; i < (count - 1); ++i) {
        qualifiers[i + 1] = qualifiers[i];
      }
      qualifiers[0] = qualifier;
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
