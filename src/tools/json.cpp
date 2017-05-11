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

#include <cstring>

#include "occa/defines.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/json.hpp"

namespace occa {
  const char json::objectKeyEndChars[] = " \t\r\n\v\f:";

  json& json::clear() {
    type = none_;
    value_.string = "";
    value_.number = 0;
    value_.object.clear();
    value_.array.clear();
    value_.boolean = false;
    return *this;
  }

  json& json::operator = (const json &j) {
    type = j.type;
    value_ = j.value_;
    return *this;
  }

  bool json::isInitialized() {
    return (type != none_);
  }

  json& json::load(const char *&c) {
    clear();
    lex::skipWhitespace(c);
    switch (*c) {
    case '0': case '1': case '2': case '3': case '4':
    case '5': case '6': case '7': case '8': case '9':
    case '-': loadNumber(c); break;
    case '{': loadObject(c); break;
    case '[': loadArray(c);  break;
    case '\'':
    case '"': loadString(c); break;
    case 't': loadTrue(c);   break;
    case 'f': loadFalse(c);  break;
    case 'n': loadNull(c);   break;
    default: {
      OCCA_FORCE_ERROR("Cannot load JSON");
    }}
    return *this;
  }

  json& json::load(const std::string &s) {
    const char *c = s.c_str();
    load(c);
    return *this;
  }

  json json::loads(const std::string &filename) {
    json j;
    j.load(io::read(filename));
    return j;
  }

  void json::dumps(const std::string &filename) {
    io::write(filename, toString());
  }

  void json::loadString(const char *&c) {
    // Skip quote
    const char quote = *c;
    ++c;
    type = string_;

    while (*c != '\0') {
      if (*c == '\\') {
        ++c;
        switch (*c) {
        case '"':  value_.string += '"';  break;
        case '\\': value_.string += '\\'; break;
        case '/':  value_.string += '/';  break;
        case 'b':  value_.string += '\b'; break;
        case 'f':  value_.string += '\f'; break;
        case 'n':  value_.string += '\n'; break;
        case 'r':  value_.string += '\r'; break;
        case 't':  value_.string += '\t'; break;
        case 'u':
          OCCA_FORCE_ERROR("Unicode is not supported yet");
          break;
        default:
          OCCA_FORCE_ERROR("Cannot escape character: '" << *c << "'");
        }
        ++c;
      } else if (*c == quote) {
        ++c;
        return;
      } else {
        value_.string += *(c++);
      }
    }
    OCCA_FORCE_ERROR("Unclosed string");
  }

  void json::loadNumber(const char *&c) {
    type = number_;
    value_.number = primitive::load(c);
  }

  void json::loadObject(const char *&c) {
    // Skip {
    const char hasBrace = (*c == '{');
    if (hasBrace) {
      ++c;
    }
    type = object_;

    while (*c != '\0') {
      lex::skipWhitespace(c);
      // Trailing ,
      if ((*c == '}') ||
          (*c == '\0')) {
        break;
      }

      loadObjectField(c);
      lex::skipWhitespace(c);

      if (*c == ',') {
        ++c;
        continue;
      } else if(*c == '}') {
        break;
      } else if(*c == '\0') {
        if (hasBrace) {
          OCCA_FORCE_ERROR("Object is missing closing '}'");
        } else {
          break;
        }
      }
      OCCA_FORCE_ERROR("Object key-values should be followed by ',' or '}'");
    }

    // Skip }
    if (hasBrace) {
      ++c;
    }
  }

  void json::loadObjectField(const char *&c) {
    std::string key;
    if (*c == '"') {
      json jKey;
      jKey.loadString(c);
      key = jKey.value_.string;
    } else {
      const char *cStart = c;
      lex::skipTo(c, objectKeyEndChars);
      key = std::string(cStart, c - cStart);
    }
    OCCA_ERROR("Key cannot be of size 0",
               key.size());

    lex::skipWhitespace(c);
    OCCA_ERROR("Key must be followed by ':'",
               *c == ':');
    ++c;
    value_.object[key].load(c);
  }

  void json::loadArray(const char *&c) {
    // Skip [
    ++c;
    type = array_;

    while (*c != '\0') {
      lex::skipWhitespace(c);
      // Trailing ,
      if (*c == ']') {
        break;
      }

      value_.array.push_back(json());
      value_.array[value_.array.size() - 1].load(c);
      lex::skipWhitespace(c);

      if (*c == ',') {
        ++c;
        continue;
      } else if(*c == ']') {
        break;
      } else if(*c == '\0') {
        OCCA_FORCE_ERROR("Array is missing closing ']'");
      }
      OCCA_FORCE_ERROR("Array values should be followed by ',' or ']'");
    }

    // Skip ]
    ++c;
  }

  void json::loadTrue(const char *&c) {
    type = boolean_;
    OCCA_ERROR("Cannot read value: " << c,
               !strncmp(c, "true", 4));
    c += 4;
    *this = true;
  }

  void json::loadFalse(const char *&c) {
    type = boolean_;
    OCCA_ERROR("Cannot read value: " << c,
               !strncmp(c, "false", 5));
    c += 5;
    *this = false;
  }

  void json::loadNull(const char *&c) {
    type = null_;
    OCCA_ERROR("Cannot read value: " << c,
               !strncmp(c, "null", 4));
    c += 4;
    type = null_;
  }

  json json::operator + (const json &j) const {
    json sum = *this;
    sum += j;
    return sum;
  }

  json& json::operator += (const json &j) {
    OCCA_ERROR("Cannot apply operator + with different JSON types",
               (type == array_) ||
               (type == j.type));

    switch(type) {
    case none_: {
      break;
    }
    case string_: {
      value_.string += j.value_.string;
      break;
    }
    case number_: {
      value_.number += j.value_.number;
      break;
    }
    case object_: {
      mergeWithObject(j.value_.object);
      break;
    }
    case array_: {
      value_.array.push_back(j);
      break;
    }
    case boolean_: {
      value_.boolean += j.value_.boolean;
      break;
    }
    case null_: {
      break;
    }}
    return *this;
  }

  void json::mergeWithObject(const jsonObject_t &obj) {
    cJsonObjectIterator it = obj.begin();
    while (it != obj.end()) {
      const std::string &key = it->first;
      const json &val = it->second;
      ++it;

      // If we're merging two json objects, recursively merge them
      if (val.isObject() && has(key)) {
        // Reuse prefetch
        json &oldVal = value_.object[key];
        if (oldVal.isObject()) {
          oldVal += val;
        } else {
          oldVal = val;
        }
      } else {
        value_.object[key] = val;
      }
    }
  }

  bool json::has(const std::string &s) const {
    const char *c  = s.c_str();
    const json *j = this;

    while (*c != '\0') {
      if (j->type != object_) {
        return false;
      }

      const char *cStart = c;
      lex::skipTo(c, '/', '\\');
      std::string key(cStart, c - cStart);
      if (*c == '/') {
        ++c;
      }

      cJsonObjectIterator it = j->value_.object.find(key);
      if (it == j->value_.object.end()) {
        return false;
      }
      j = &(it->second);
    }
    return true;
  }

  json& json::operator [] (const char *c) {
     const char *c0 = c;
     json *j = this;

     if (type == none_) {
       type = object_;
     }

     while (*c != '\0') {
       OCCA_ERROR("Path '" << std::string(c0, c - c0) << "' is not an object",
                  j->type == object_);

       const char *cStart = c;
       lex::skipTo(c, '/', '\\');
       std::string key(cStart, c - cStart);
       if (*c == '/') {
         ++c;
       }

       j = &(j->value_.object[key]);
       if (j->type == none_) {
         j->type = object_;
       }
     }
     return *j;
   }

  const json& json::operator [] (const char *c) const {
    static json default_;
    const json *j = this;
    while (*c != '\0') {
      if (j->type != object_) {
        return default_;
      }

      const char *cStart = c;
      lex::skipTo(c, '/', '\\');
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

  json& json::operator [] (const int n) {
    OCCA_ERROR("Can only apply operator [] with JSON arrays",
               type == array_);
    return value_.array[n];
  }

  const json& json::operator [] (const int n) const {
    OCCA_ERROR("Can only apply operator [] with JSON arrays",
               type == array_);
    return value_.array[n];
  }

  json& json::remove(const char *c) {
    json *j = this;
    while (*c != '\0') {
      if (j->type != object_) {
        return *this;
      }

      const char *cStart = c;
      lex::skipTo(c, '/', '\\');
      std::string key(cStart, c - cStart);
      if (*c == '/') {
        ++c;
      }

      if (*c == '\0') {
        j->value_.object.erase(key);
        return *this;
      }

      jsonObjectIterator it = j->value_.object.find(key);
      if (it == j->value_.object.end()) {
        return *this;
      }
      j = &(it->second);
    }
    return *this;
  }

  hash_t json::hash() const {
    std::string out;
    toString(out);
    return occa::hash(out);
  }

  std::string json::toString() const {
    std::string out;
    toString(out);
    return out;
  }

  void json::toString(std::string &out, const std::string &indent) const {
    switch(type) {
    case none_: {
      return;
    }
    case string_: {
      out += '"';
      const int chars = (int) value_.string.size();
      for (int i = 0; i < chars; ++i) {
        const char c = value_.string[i];
        switch (c) {
        case '"' : out += "\\\"";  break;
        case '\\': out += "\\\\";  break;
        case '/' : out += "\\/";   break;
        case '\b': out += "\\b";  break;
        case '\f': out += "\\f";  break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
        default:
          out += c;
        }
      }
      out += '"';
      break;
    }
    case number_: {
      out += (std::string) value_.number;
      break;
    }
    case object_: {
      cJsonObjectIterator it = value_.object.begin();
      out += '{';
      if (it != value_.object.end()) {
        std::string newIndent = indent + "  ";
        out += '\n';
        while (it != value_.object.end()) {
          out += newIndent;
          out += '"';
          out += it->first;
          out += "\": ";
          it->second.toString(out, newIndent);
          out += ",\n";
          ++it;
        }
      }
      out += indent;
      out += '}';
      break;
    }
    case array_: {
      out += '[';
      const int arraySize = (int) value_.array.size();
      if (arraySize) {
        std::string newIndent = indent + "  ";
        out += '\n';
        for (int i = 0; i < arraySize; ++i) {
          out += newIndent;
          value_.array[i].toString(out, newIndent);
          out += ",\n";
        }
        out += indent;
      }
      out += ']';
      break;
    }
    case boolean_: {
      out += value_.boolean ? "true" : "false";
      break;
    }
    case null_: {
      out += "null";
    }}
  }

  template <>
  hash_t hash(const occa::json &json) {
    return json.hash();
  }

  std::ostream& operator << (std::ostream &out, const json &j) {
    if (j.isString()) {
      out << j.string();
    } else {
      out << j.toString() << '\n';
    }
    return out;
  }
}
