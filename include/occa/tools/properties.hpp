/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2016 David Medina and Tim Warburton
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

#ifndef OCCA_TOOLS_PROPERTIES_HEADER
#define OCCA_TOOLS_PROPERTIES_HEADER

#include "occa/defines.hpp"
#include "occa/types.hpp"
#include "occa/tools/string.hpp"
#include "occa/tools/hash.hpp"

namespace occa {
  class hasProperties;

  //---[ properties ]-------------------
  class properties {
  public:
    enum Op {
      Set, Remove
    };

    typedef strToStrMapIterator  iter_t;
    typedef cStrToStrMapIterator citer_t;

  protected:
    static void *_NULL;
    strToStrMap_t props;
    hasProperties *holder;

  public:
    properties(hasProperties *holder_ = NULL);
    properties(const std::string &props_);
    properties(const char *props_);

    properties(const properties &p);
    properties& operator = (const properties &p);

    void initFromString(const std::string &props_);

    udim_t size();

    bool has(const std::string &prop) const;

    std::string& operator [] (const std::string &prop);
    const std::string operator [] (const std::string &prop) const;

    template <class TM>
    TM get(const std::string &prop, const TM &default_ = TM()) const {
      citer_t it = props.find(prop);
      if (it != props.end()) {
        return occa::fromString<TM>(it->second);
      }
      return default_;
    }

    template <class TM*>
    TM* get(const std::string &prop, const TM *&default_ = (const TM*&) _NULL) const {
      return (TM*) get<uintptr_t>(prop, (uintptr_t) default_);
    }

    template <class TM>
    std::vector<TM> getList(const std::string &prop, const std::vector<TM> &default_ = std::vector<TM>()) const {
      citer_t it = props.find(prop);
      if (it == props.end()) {
        return default_;
      }
      return listFromString<TM>(it->second);
    }

    std::string get(const std::string &prop, const std::string &default_ = "") const;

    template <class TM>
    properties& set(const std::string &prop, const TM &t) {
      iter_t it = props.find(prop);
      std::string newValue = occa::toString(t);

      if (it != props.end()) {
        std::string &oldValue = it->second;
        if (oldValue != newValue) {
          onChange(Set, prop, oldValue, newValue);
          it->second = newValue;
        }
      } else {
        props[prop] = newValue;
        onChange(Set, prop, "", newValue);
      }
      return *this;
    }

    template <class TM*>
    properties& set(const std::string &prop, const TM* &t) {
      return set<uintptr_t>(prop, (uintptr_t) t);
    }

    template <class TM>
    properties& setIfMissing(const std::string &prop, const TM &t) {
      if (!has(prop)) {
        set(prop, t);
      }
      return *this;
    }

    properties operator + (const properties &props) const;

    void remove(const std::string &prop);

    bool iMatch(const std::string &prop, const std::string &value) const;

    void setOnChangeFunc(hasProperties &holder_);
    void onChange(properties::Op op,
                  const std::string &prop,
                  const std::string &oldValue,
                  const std::string &newValue) const;

    hash_t hash() const;

    std::string toString() const;
    friend std::ostream& operator << (std::ostream &out, const properties &props);
  };
  std::ostream& operator << (std::ostream &out, const properties &props);
  //====================================

  //---[ hasProperties ]----------------
  class hasProperties {
  public:
    occa::properties properties;

    hasProperties();

    void onPropertyChange(properties::Op op,
                          const std::string &prop,
                          const std::string &oldValue,
                          const std::string &newValue) const;
  };
  //====================================
}

#endif
