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

  json::json(type_t type_) {
    clear();
    type = type_;
  }

  json::json(const json &j) :
    type(j.type),
    value(j.value) {}

  json& json::clear() {
    type = none_;
    value.string = "";
    value.number = 0;
    value.object.clear();
    value.array.clear();
    value.boolean = false;
    return *this;
  }

  json& json::operator = (const json &j) {
    type = j.type;
    value = j.value;
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
    return json(io::read(filename));
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
        case '"':  value.string += '"';  break;
        case '\\': value.string += '\\'; break;
        case '/':  value.string += '/';  break;
        case 'b':  value.string += '\b'; break;
        case 'f':  value.string += '\f'; break;
        case 'n':  value.string += '\n'; break;
        case 'r':  value.string += '\r'; break;
        case 't':  value.string += '\t'; break;
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
        value.string += *(c++);
      }
    }
    OCCA_FORCE_ERROR("Unclosed string");
  }

  void json::loadNumber(const char *&c) {
    type = number_;
    value.number = primitive::load(c);
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
      key = jKey.value.string;
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
    value.object[key].load(c);
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

      value.array.push_back(json());
      value.array[value.array.size() - 1].load(c);
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

  json json::operator + (const json &j) {
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
      value.string += j.value.string;
      break;
    }
    case number_: {
      value.number += j.value.number;
      break;
    }
    case object_: {
      value.object.insert(j.value.object.begin(),
                          j.value.object.end());
      break;
    }
    case array_: {
      value.array.push_back(j);
      break;
    }
    case boolean_: {
      value.boolean += j.value.boolean;
      break;
    }
    case null_: {
      break;
    }}
    return *this;
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

      cJsonObjectIterator it = j->value.object.find(key);
      if (it == j->value.object.end()) {
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

       j = &(j->value.object[key]);
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

      cJsonObjectIterator it = j->value.object.find(key);
      if (it == j->value.object.end()) {
        return default_;
      }
      j = &(it->second);
    }
    return *j;
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
        j->value.object.erase(key);
        return *this;
      }

      jsonObjectIterator it = j->value.object.find(key);
      if (it == j->value.object.end()) {
        return *this;
      }
      j = &(it->second);
    }
    return *this;
  }

  hash_t json::hash() const {
    hash_t hash_;
    switch(type) {
    case none_: {
      break;
    }
    case string_: {
      hash_ ^= occa::hash(value.string);
      break;
    }
    case number_: {
      hash_ ^= occa::hash(value.number.value.ptr);
      break;
    }
    case object_: {
      cJsonObjectIterator it = value.object.begin();
      while (it != value.object.end()) {
        hash_ ^= occa::hash(it->first);
        hash_ ^= occa::hash(it->second);
        ++it;
      }
      break;
    }
    case array_: {
      const int arraySize = (int) value.array.size();
      for (int i = 0; i < arraySize; ++i) {
        hash_ ^= occa::hash(value.array[i]);
      }
      break;
    }
    case boolean_: {
      hash_ ^= occa::hash(value.boolean);
      break;
    }
    case null_: {
      hash_ ^= occa::hash("null");
      break;
    }}
    return hash_;
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
      const int chars = (int) value.string.size();
      for (int i = 0; i < chars; ++i) {
        const char c = value.string[i];
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
      out += (std::string) value.number;
      break;
    }
    case object_: {
      cJsonObjectIterator it = value.object.begin();
      out += '{';
      if (it != value.object.end()) {
        std::string newIndent = indent + "  ";
        out += '\n';
        while (it != value.object.end()) {
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
      const int arraySize = (int) value.array.size();
      if (arraySize) {
        std::string newIndent = indent + "  ";
        out += '\n';
        for (int i = 0; i < arraySize; ++i) {
          out += newIndent;
          value.array[i].toString(out, newIndent);
          out += ",\n";
        }
        out += indent;
      }
      out += ']';
      break;
    }
    case boolean_: {
      out += value.boolean ? "true" : "false";
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
    out << j.toString() << '\n';
    return out;
  }
}
