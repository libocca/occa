/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#ifndef OCCA_TOOLS_JSON_HEADER
#define OCCA_TOOLS_JSON_HEADER

#include <vector>
#include <map>

#include "occa/parser/primitive.hpp"
#include "occa/tools/hash.hpp"
#include "occa/tools/lex.hpp"

namespace occa {
  class json;

  typedef std::map<std::string, json>  jsonObject_t;
  typedef jsonObject_t::iterator       jsonObjectIterator;
  typedef jsonObject_t::const_iterator cJsonObjectIterator;

  typedef std::vector<json> jsonArray_t;

  typedef struct {
    std::string string;
    primitive number;
    jsonObject_t object;
    jsonArray_t array;
    bool boolean;
  } jsonValue_t;

  class json {
  public:
    static const char objectKeyEndChars[];

    enum type_t {
      none_    = (1 << 0),
      string_  = (1 << 1),
      number_  = (1 << 2),
      object_  = (1 << 3),
      array_   = (1 << 4),
      boolean_ = (1 << 5),
      null_    = (1 << 6)
    };

    type_t type;
    jsonValue_t value_;

    inline json(type_t type_ = none_) {
      clear();
      type = type_;
    }

    inline json(const json &j) :
      type(j.type),
      value_(j.value_) {}

    inline json(const bool value) :
      type(boolean_) {
      value_.boolean = value;
    }

    inline json(const uint8_t value) :
      type(number_) {
      value_.number = value;
    }

    inline json(const int8_t value) :
      type(number_) {
      value_.number = value;
    }

    inline json(const uint16_t value) :
      type(number_) {
      value_.number = value;
    }

    inline json(const int16_t value) :
      type(number_) {
      value_.number = value;
    }

    inline json(const uint32_t value) :
      type(number_) {
      value_.number = value;
    }

    inline json(const int32_t value) :
      type(number_) {
      value_.number = value;
    }

    inline json(const uint64_t value) :
      type(number_) {
      value_.number = value;
    }

    inline json(const int64_t value) :
      type(number_) {
      value_.number = value;
    }

    inline json(const double value) :
      type(number_) {
      value_.number = value;
    }

    inline json(const primitive &value) :
      type(number_) {
      value_.number = value;
    }

    inline json(const std::string &value) :
      type(string_) {
      value_.string = value;
    }

    inline json(const jsonObject_t &value) :
      type(object_) {
      value_.object = value;
    }

    inline json(const jsonArray_t &value) :
      type(array_) {
      value_.array = value;
    }

    json& clear();

    json& operator = (const json &j);

    inline json& operator = (const char *c) {
      type = string_;
      value_.string = c;
      return *this;
    }

    inline json& operator = (const std::string &value) {
      type = string_;
      value_.string = value;
      return *this;
    }

    inline json& operator = (const bool value) {
      type = boolean_;
      value_.boolean = value;
      return *this;
    }

    inline json& operator = (const uint8_t value) {
      type = number_;
      value_.number = value;
      return *this;
    }

    inline json& operator = (const int8_t value) {
      type = number_;
      value_.number = value;
      return *this;
    }

    inline json& operator = (const uint16_t value) {
      type = number_;
      value_.number = value;
      return *this;
    }

    inline json& operator = (const int16_t value) {
      type = number_;
      value_.number = value;
      return *this;
    }

    inline json& operator = (const uint32_t value) {
      type = number_;
      value_.number = value;
      return *this;
    }

    inline json& operator = (const int32_t value) {
      type = number_;
      value_.number = value;
      return *this;
    }

    inline json& operator = (const uint64_t value) {
      type = number_;
      value_.number = value;
      return *this;
    }

    inline json& operator = (const int64_t value) {
      type = number_;
      value_.number = value;
      return *this;
    }

    inline json& operator = (const double value) {
      type = number_;
      value_.number = value;
      return *this;
    }

    inline json& operator = (const primitive &value) {
      type = number_;
      value_.number = value;
      return *this;
    }

    inline json& operator = (const jsonObject_t &value) {
      type = object_;
      value_.object = value;
      return *this;
    }

    inline json& operator = (const jsonArray_t &value) {
      type = array_;
      value_.array = value;
      return *this;
    }

    bool isInitialized();

    json& load(const char *&c);
    json& load(const std::string &s);

    static json loads(const std::string &filename);
    void dumps(const std::string &filename);

    void loadString(const char *&c);
    void loadNumber(const char *&c);
    void loadObject(const char *&c);
    void loadObjectField(const char *&c);
    void loadArray(const char *&c);
    void loadTrue(const char *&c);
    void loadFalse(const char *&c);
    void loadNull(const char *&c);

    json operator + (const json &j) const;
    json& operator += (const json &j);

    void mergeWithObject(const jsonObject_t &obj);

    bool has(const std::string &s) const;

    inline bool isString() const {
      return (type == string_);
    }

    inline bool isNumber() const {
      return (type == number_);
    }

    inline bool isObject() const {
      return (type == object_);
    }

    inline bool isArray() const {
      return (type == array_);
    }

    inline bool isBoolean() const {
      return (type == boolean_);
    }

    inline json& asString() {
      type = string_;
      return *this;
    }

    inline json& asNumber() {
      type = number_;
      return *this;
    }

    inline json& asObject() {
      type = object_;
      return *this;
    }

    inline json& asArray() {
      type = array_;
      return *this;
    }

    inline json& asBoolean() {
      type = boolean_;
      return *this;
    }

    inline std::string& string() {
      return value_.string;
    }

    inline primitive& number() {
      return value_.number;
    }

    inline jsonObject_t& object() {
      return value_.object;
    }

    inline jsonArray_t& array() {
      return value_.array;
    }

    inline bool& boolean() {
      return value_.boolean;
    }

    inline const std::string& string() const {
      return value_.string;
    }

    inline const primitive& number() const {
      return value_.number;
    }

    inline const jsonObject_t& object() const {
      return value_.object;
    }

    inline const jsonArray_t& array() const {
      return value_.array;
    }

    inline bool boolean() const {
      return value_.boolean;
    }

    json& operator [] (const char *c);
    const json& operator [] (const char *c) const;

    inline json& operator [] (const std::string &s) {
      return (*this)[s.c_str()];
    }

    inline const json& operator [] (const std::string &s) const {
      return (*this)[s.c_str()];
    }

    json& operator [] (const int n);
    const json& operator [] (const int n) const;

    int size() const;

    template <class TM>
    TM get(const char *c, const TM &default_ = TM()) const {
      const char *c0 = c;
      const json *j = this;
      while (*c != '\0') {
        OCCA_ERROR("Path '" << std::string(c0, c - c0) << "' is not an object",
                   j->type == object_);

        const char *cStart = c;
        lex::skipTo(c, '/');
        std::string key(cStart, c - cStart);
        if (*c == '/') {
          ++c;
        }

        cJsonObjectIterator it = j->value_.object.find(key);
        if (it == j->value_.object.end()) {
          return default_;
        }
        j = &(it->second);
      }
      return *j;
    }

    template <class TM>
    inline TM get(const std::string &s, const TM &default_ = TM()) const {
      return get<TM>(s.c_str(), default_);
    }

    template <class TM>
    std::vector<TM> getArray(const std::vector<TM> &default_ = std::vector<TM>()) const {
      std::string empty;
      return getArray(empty.c_str(), default_);
    }

    friend std::ostream& operator << (std::ostream &out, const json &j);

    template <class TM>
    std::vector<TM> getArray(const char *c,
                             const std::vector<TM> &default_ = std::vector<TM>()) const {
      const char *c0 = c;
      const json *j = this;
      while (*c) {
        OCCA_ERROR("Path '" << std::string(c0, c - c0) << "' is not an object",
                   j->type == object_);

        const char *cStart = c;
        lex::skipTo(c, '/');
        std::string key(cStart, c - cStart);
        if (*c == '/') {
          ++c;
        }

        cJsonObjectIterator it = j->value_.object.find(key);
        if (it == j->value_.object.end()) {
          return default_;
        }
        j = &(it->second);
      }
      if (j->type != array_) {
        return default_;
      }

      const int entries = (int) j->value_.array.size();
      std::vector<TM> ret;
      for (int i = 0; i < entries; ++i) {
        ret.push_back((TM) j->value_.array[i]);
      }
      return ret;
    }

    template <class TM>
    std::vector<TM> getArray(const std::string &s,
                             const std::vector<TM> &default_ = std::vector<TM>()) const {
      return get<TM>(s.c_str(), default_);
    }

    json& remove(const char *c);

    inline json& remove(const std::string &s) {
      remove(s.c_str());
      return *this;
    }

    inline bool operator == (const json &j) const {
      if (type != j.type) {
        return false;
      }
      switch(type) {
      case none_:
        return true;
      case string_:
        return value_.string == j.value_.string;
      case number_:
        return value_.number == j.value_.number;
      case object_:
        return value_.object == j.value_.object;
      case array_:
        return value_.array == j.value_.array;
      case boolean_:
        return value_.boolean == j.value_.boolean;
      case null_:
        return true;
      default:
        return false;
      }
    }

    inline operator bool () const {
      return value_.boolean;
    }

    inline operator uint8_t () const {
      return (uint8_t) value_.number;
    }

    inline operator uint16_t () const {
      return (uint16_t) value_.number;
    }

    inline operator uint32_t () const {
      return (uint32_t) value_.number;
    }

    inline operator uint64_t () const {
      return (uint64_t) value_.number;
    }

    inline operator int8_t () const {
      return (int8_t) value_.number;
    }

    inline operator int16_t () const {
      return (int16_t) value_.number;
    }

    inline operator int32_t () const {
      return (int32_t) value_.number;
    }

    inline operator int64_t () const {
      return (int64_t) value_.number;
    }

    inline operator float () const {
      return (float) value_.number;
    }

    inline operator double () const {
      return (double) value_.number;
    }

    inline operator std::string () const {
      return value_.string;
    }

    hash_t hash() const;

    std::string toString() const;
    void toString(std::string &out, const std::string &indent = "") const;
  };

  template <>
  hash_t hash(const occa::json &json);

  std::ostream& operator << (std::ostream &out, const json &j);
}

#endif
