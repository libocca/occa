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

    struct {
      std::string string;
      primitive number;
      jsonObject_t object;
      jsonArray_t array;
      bool boolean;
    } value;

    json(type_t type_ = none_);
    json(const json &j);

    inline json(const bool boolean) :
      type(boolean_) {
      value.boolean = boolean;
    }

    inline json(const uint8_t number) :
      type(number_) {
      value.number = number;
    }

    inline json(const int8_t number) :
      type(number_) {
      value.number = number;
    }

    inline json(const uint16_t number) :
      type(number_) {
      value.number = number;
    }

    inline json(const int16_t number) :
      type(number_) {
      value.number = number;
    }

    inline json(const uint32_t number) :
      type(number_) {
      value.number = number;
    }

    inline json(const int32_t number) :
      type(number_) {
      value.number = number;
    }

    inline json(const uint64_t number) :
      type(number_) {
      value.number = number;
    }

    inline json(const int64_t number) :
      type(number_) {
      value.number = number;
    }

    inline json(const double number) :
      type(number_) {
      value.number = number;
    }

    inline json(const primitive &number) :
      type(number_) {
      value.number = number;
    }

    inline json(const std::string &string) :
      type(string_) {
      value.string = string;
    }

    inline json(const jsonObject_t &object) :
      type(object_) {
      value.object = object;
    }

    inline json(const jsonArray_t &array) :
      type(array_) {
      value.array = array;
    }

    json& clear();

    json& operator = (const json &j);

    inline json& operator = (const std::string &string) {
      type = string_;
      value.string = string;
      return *this;
    }

    inline json& operator = (const bool boolean) {
      type = boolean_;
      value.boolean = boolean;
      return *this;
    }

    inline json& operator = (const uint8_t number) {
      type = number_;
      value.number = number;
      return *this;
    }

    inline json& operator = (const int8_t number) {
      type = number_;
      value.number = number;
      return *this;
    }

    inline json& operator = (const uint16_t number) {
      type = number_;
      value.number = number;
      return *this;
    }

    inline json& operator = (const int16_t number) {
      type = number_;
      value.number = number;
      return *this;
    }

    inline json& operator = (const uint32_t number) {
      type = number_;
      value.number = number;
      return *this;
    }

    inline json& operator = (const int32_t number) {
      type = number_;
      value.number = number;
      return *this;
    }

    inline json& operator = (const uint64_t number) {
      type = number_;
      value.number = number;
      return *this;
    }

    inline json& operator = (const int64_t number) {
      type = number_;
      value.number = number;
      return *this;
    }

    inline json& operator = (const double number) {
      type = number_;
      value.number = number;
      return *this;
    }

    inline json& operator = (const primitive &number) {
      type = number_;
      value.number = number;
      return *this;
    }

    inline json& operator = (const jsonObject_t &object) {
      type = object_;
      value.object = object;
      return *this;
    }

    inline json& operator = (const jsonArray_t &array) {
      type = array_;
      value.array = array;
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

    json operator + (const json &j);
    json& operator += (const json &j);

    bool has(const std::string &s) const;

    inline bool isString() {
      return (type == string_);
    }

    inline bool isNumber() {
      return (type == number_);
    }

    inline bool isObject() {
      return (type == object_);
    }

    inline bool isArray() {
      return (type == array_);
    }

    inline bool isBoolean() {
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
      return value.string;
    }

    inline primitive& number() {
      return value.number;
    }

    inline jsonObject_t& object() {
      return value.object;
    }

    inline jsonArray_t& array() {
      return value.array;
    }

    inline bool& boolean() {
      return value.boolean;
    }

    inline const std::string& string() const {
      return value.string;
    }

    inline const primitive& number() const {
      return value.number;
    }

    inline const jsonObject_t& object() const {
      return value.object;
    }

    inline const jsonArray_t& array() const {
      return value.array;
    }

    inline bool boolean() const {
      return value.boolean;
    }

    json& operator [] (const char *c);
    const json& operator [] (const char *c) const;

    inline json& operator [] (const std::string &s) {
      return (*this)[s.c_str()];
    }

    inline const json& operator [] (const std::string &s) const {
      return (*this)[s.c_str()];
    }

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

        cJsonObjectIterator it = j->value.object.find(key);
        if (it == j->value.object.end()) {
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

        cJsonObjectIterator it = j->value.object.find(key);
        if (it == j->value.object.end()) {
          return default_;
        }
        j = &(it->second);
      }
      if (j->type != array_) {
        return default_;
      }

      const int entries = (int) j->value.array.size();
      std::vector<TM> ret;
      for (int i = 0; i < entries; ++i) {
        ret.push_back((TM) j->value.array[i]);
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
        return value.string == j.value.string;
      case number_:
        return value.number == j.value.number;
      case object_:
        return value.object == j.value.object;
      case array_:
        return value.array == j.value.array;
      case boolean_:
        return value.boolean == j.value.boolean;
      case null_:
        return true;
      default:
        return false;
      }
    }

    inline operator bool () const {
      return value.boolean;
    }

    inline operator uint8_t () const {
      return (uint8_t) value.number;
    }

    inline operator uint16_t () const {
      return (uint16_t) value.number;
    }

    inline operator uint32_t () const {
      return (uint32_t) value.number;
    }

    inline operator uint64_t () const {
      return (uint64_t) value.number;
    }

    inline operator int8_t () const {
      return (int8_t) value.number;
    }

    inline operator int16_t () const {
      return (int16_t) value.number;
    }

    inline operator int32_t () const {
      return (int32_t) value.number;
    }

    inline operator int64_t () const {
      return (int64_t) value.number;
    }

    inline operator float () const {
      return (float) value.number;
    }

    inline operator double () const {
      return (double) value.number;
    }

    inline operator std::string () const {
      return value.string;
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
