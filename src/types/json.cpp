#include <cstring>

#include <occa/defines.hpp>
#include <occa/internal/io.hpp>
#include <occa/types/json.hpp>
#include <occa/internal/utils/lex.hpp>

namespace occa {
  const char json::objectKeyEndChars[] = " \t\r\n\v\f:";

  json::json(const std::string &name,
             const primitive &value) {
    (*this)[name] = value;
  }

  json::json(std::initializer_list<jsonKeyValue> entries) {
    type = object_;
    for (auto &entry : entries) {
      (*this)[entry.name] = entry.value;
    }
  }

  json::~json() {}

  json& json::clear() {
    type = none_;
    value_.string = "";
    value_.number = 0;
    value_.object.clear();
    value_.array.clear();
    return *this;
  }

  json& json::operator = (const json &j) {
    type = j.type;
    value_ = j.value_;
    return *this;
  }

  bool json::isInitialized() const {
    return (type != none_);
  }

  json& json::load(const char *&c) {
    clear();
    lex::skipWhitespace(c);
    switch (*c) {
      case '0': case '1': case '2': case '3': case '4':
      case '5': case '6': case '7': case '8': case '9':
      case '-': loadNumber(c);  break;
      case '{': loadObject(c);  break;
      case '[': loadArray(c);   break;
      case '\'':
      case '"': loadString(c);  break;
      case 't': loadTrue(c);    break;
      case 'f': loadFalse(c);   break;
      case 'n': loadNull(c);    break;
      case '/': loadComment(c); break;
      default: {
        OCCA_FORCE_ERROR("Cannot load JSON: " << c);
      }}
    return *this;
  }

  json& json::load(const std::string &s) {
    const char *c = s.c_str();
    load(c);
    return *this;
  }

  std::string json::dump(const int indent) const {
    const int indent_ = indent >= 0 ? indent : 2;

    std::string out;
    std::string indentStr(indent_, ' ');
    dumpToString(out, indentStr);
    return out;
  }

  void json::dumpToString(std::string &out,
                          const std::string &indent,
                          const std::string &currentIndent) const {
    switch(type) {
    case none_: {
      return;
    }
    case null_: {
      out += "null";
      break;
    }
    case number_: {
      out += value_.number.toString();
      break;
    }
    case string_: {
      out += '"';
      const int chars = (int) value_.string.size();
      for (int i = 0; i < chars; ++i) {
        const char c = value_.string[i];
        switch (c) {
        case '"' : out += "\\\"";  break;
        case '\\': out += "\\\\";  break;
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
    case array_: {
      out += '[';
      const int arraySize = (int) value_.array.size();
      if (arraySize) {
        std::string newIndent = currentIndent + indent;
        if (indent.size()) {
          out += '\n';
        }
        for (int i = 0; i < arraySize; ++i) {
          out += newIndent;
          value_.array[i].dumpToString(out, indent, newIndent);
          if (i < (arraySize - 1)) {
            if (indent.size()) {
              out += ",\n";
            } else {
              out += ", ";
            }
          } else if (indent.size()) {
            out += '\n';
          }
        }
        out += currentIndent;
      }
      out += ']';
      break;
    }
    case object_: {
      if (!value_.object.size()) {
        out += "{}";
        break;
      }
      jsonObject::const_iterator it = value_.object.begin();
      out += '{';
      if (it != value_.object.end()) {
        std::string newIndent = currentIndent + indent;
        if (indent.size()) {
          out += '\n';
        }
        while (it != value_.object.end()) {
          const std::string &key = it->first;
          const json &value = it->second;

          out += newIndent;
          out += '"';
          out += key;
          out += "\": ";
          if (value.type != none_) {
            value.dumpToString(out, indent, newIndent);
          } else {
            // Temporary until jsonRef
            out += "{}";
          }

          ++it;
          if (it != value_.object.end()) {
            if (indent.size()) {
              out += ",\n";
            } else {
              out += ", ";
            }
          } else if (indent.size()) {
            out += '\n';
          }
        }
      }
      if (indent.size()) {
        out += currentIndent;
      }
      out += '}';
      break;
    }}
  }

  json json::parse(const char *&c) {
    json j;
    j.load(c);
    return j;
  }

  json json::parse(const std::string &s) {
    json j;
    j.load(s);
    return j;
  }

  json json::read(const std::string &filename) {
    json j;
    j.load(io::read(filename));
    return j;
  }

  void json::write(const std::string &filename) {
    io::write(filename, dump());
  }

  void json::loadString(const char *&c) {
    // Skip quote
    const char quote = *c;
    ++c;
    type = string_;

    while (*c != '\0') {
      if (*c == '\\') {
        ++c; // Skip '\'
        OCCA_ERROR("Unclosed string",
                   *c != '\0');

        switch (*c) {
          // Escape newline character
        case '\n': ++c; continue;
        case 'b':  value_.string += '\b'; break;
        case 'f':  value_.string += '\f'; break;
        case 'n':  value_.string += '\n'; break;
        case 'r':  value_.string += '\r'; break;
        case 't':  value_.string += '\t'; break;
        case 'u':
          // Found unicode character
          // Load \uXXXX
          ++c; // Skip 'u'
          value_.string += "\\u";
          for (int i = 0; i < 4; ++i) {
            const char ci = c[i];
            OCCA_ERROR("Expected hex value",
                       (('0' <= ci) && (ci <= '9')) ||
                       (('a' <= ci) && (ci <= 'f')) ||
                       (('A' <= ci) && (ci <= 'F')));
            value_.string += ci;
          }
          // Let the ++c increment the last character
          c += 3;
          break;
        default:
          value_.string += *c;
        }
        // Skip the last used character
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
        ++c; // Skip ]
        return;
      }

      value_.array.push_back(json());
      value_.array[value_.array.size() - 1].load(c);
      lex::skipWhitespace(c);

      if (*c == ',') {
        ++c;
        continue;
      } else if(*c == ']') {
        ++c; // Skip ]
        return;
      } else if(*c == '\0') {
        break;
      }
      OCCA_FORCE_ERROR("Array values should be followed by ',' or ']'");
    }
    OCCA_FORCE_ERROR("Array is missing closing ']'");
  }

  void json::loadTrue(const char *&c) {
    OCCA_ERROR("Cannot read value: " << c,
               !strncmp(c, "true", 4));
    c += 4;
    type = number_;
    value_.number = true;
  }

  void json::loadFalse(const char *&c) {
    OCCA_ERROR("Cannot read value: " << c,
               !strncmp(c, "false", 5));
    c += 5;
    type = number_;
    value_.number = false;
  }

  void json::loadNull(const char *&c) {
    OCCA_ERROR("Cannot read value: " << c,
               !strncmp(c, "null", 4));
    c += 4;
    type = null_;
  }

  void json::loadComment(const char *&c) {
    OCCA_ERROR("Cannot read value: " << c,
               !strncmp(c, "//", 2));
    lex::skipTo(c, '\n', '\\');
  }

  json json::operator + (const json &j) const {
    json sum = *this;
    sum += j;
    return sum;
  }

  json& json::operator += (const json &j) {
    // Nothing to add
    if (j.type == none_) {
      return *this;
    }

    // We're not defined, treat this as an = operator
    if (type == none_) {
      type = j.type;
    }
    OCCA_ERROR("Cannot apply operator + with different JSON types",
               (type == array_) ||
               (type == j.type));

    switch(type) {
    case none_: break;
    case null_: break;
    case number_: {
      primitive::addEq(value_.number, j.value_.number);
      break;
    }
    case string_: {
      value_.string += j.value_.string;
      break;
    }
    case array_: {
      value_.array.push_back(j);
      break;
    }
    case object_: {
      mergeWithObject(j.value_.object);
      break;
    }}
    return *this;
  }

  void json::mergeWithObject(const jsonObject &obj) {
    jsonObject::const_iterator it = obj.begin();
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

      jsonObject::const_iterator it = j->value_.object.find(key);
      if (it == j->value_.object.end()) {
        return false;
      }
      j = &(it->second);
    }
    return true;
  }

  json& json::operator [] (const char *c) {
#if !OCCA_UNSAFE
    const char *c0 = c;
#endif
    json *j = this;
    bool exists = true;

    if (type == none_) {
      type = object_;
      exists = false;
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
        exists = false;
      }
    }
    if (!exists) {
      j->type = none_;
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

      jsonObject::const_iterator it = j->value_.object.find(key);
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
    const int arraySize = (int) value_.array.size();
    if (arraySize <= n) {
      value_.array.resize(n + 1);
      for (int i = arraySize; i < n; ++i) {
        value_.array[i].asNull();
      }
    }
    return value_.array[n];
  }

  const json& json::operator [] (const int n) const {
    OCCA_ERROR("Can only apply operator [] with JSON arrays",
               type == array_);
    return value_.array[n];
  }

  int json::size() const {
    switch(type) {
    case none_: {
      return 0;
    }
    case null_: {
      return 0;
    }
    case number_: {
      return 0;
    }
    case string_: {
      return (int) value_.string.size();
    }
    case array_: {
      return (int) value_.array.size();
    }
    case object_: {
      return (int) value_.object.size();
    }}
    return 0;
  }

  json json::getPathValue(const char *key) const {
    const json *j = this;
    const char *c = key;

    while (*c != '\0') {
      if (j->type != object_) {
        return json();
      }

      const char *cStart = c;
      lex::skipTo(c, '/');
      std::string nextKey(cStart, c - cStart);
      if (*c == '/') {
        ++c;
      }

      jsonObject::const_iterator it = j->value_.object.find(nextKey);
      if (it == j->value_.object.end()) {
        return json();
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
        j->value_.object.erase(key);
        return *this;
      }

      jsonObject::iterator it = j->value_.object.find(key);
      if (it == j->value_.object.end()) {
        return *this;
      }
      j = &(it->second);
    }
    return *this;
  }

  hash_t json::hash() const {
    std::string out;
    dumpToString(out);
    return occa::hash(out);
  }

  std::string json::toString() const {
    if (type == string_) {
      return value_.string;
    }
    return dump();
  }

  strVector json::keys() const {
    strVector vec;
    if (type == object_) {
      const jsonObject &obj = value_.object;
      jsonObject::const_iterator it = obj.begin();
      while (it != obj.end()) {
        vec.push_back(it->first);
        ++it;
      }
    }
    return vec;
  }

  jsonArray json::values() const {
    jsonArray vec;
    if (type == object_) {
      const jsonObject &obj = value_.object;
      jsonObject::const_iterator it = obj.begin();
      while (it != obj.end()) {
        vec.push_back(it->second);
        ++it;
      }
    }
    return vec;
  }

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             const bool value_) :
    name(name_),
    value(value_) {}

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             const uint8_t value_) :
    name(name_),
    value(value_) {}

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             const int8_t value_) :
    name(name_),
    value(value_) {}

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             const uint16_t value_) :
    name(name_),
    value(value_) {}

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             const int16_t value_) :
    name(name_),
    value(value_) {}

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             const uint32_t value_) :
    name(name_),
    value(value_) {}

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             const int32_t value_) :
    name(name_),
    value(value_) {}

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             const uint64_t value_) :
    name(name_),
    value(value_) {}

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             const int64_t value_) :
    name(name_),
    value(value_) {}

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             const float value_) :
    name(name_),
    value(value_) {}

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             const double value_) :
    name(name_),
    value(value_) {}

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             const primitive &value_) :
    name(name_),
    value(value_) {}

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             const char *value_) :
    name(name_),
    value(value_) {}

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             const std::string &value_) :
    name(name_),
    value(value_.c_str()) {}

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             const hash_t &value_) :
    name(name_),
    value(value_.getFullString()) {}

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             const json &value_) :
    name(name_),
    value(value_) {}

  jsonKeyValue::jsonKeyValue(const std::string &name_,
                             std::initializer_list<jsonKeyValue> value_) :
    name(name_),
    value(value_) {}

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
