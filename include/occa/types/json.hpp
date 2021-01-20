#ifndef OCCA_UTILS_JSON_HEADER
#define OCCA_UTILS_JSON_HEADER

#include <map>
#include <vector>

#include <occa/dtype/builtins.hpp>
#include <occa/types/primitive.hpp>
#include <occa/utils/hash.hpp>

namespace occa {
  class json;
  class jsonKeyValue;

  typedef std::map<std::string, json> jsonObject;
  typedef std::vector<json>           jsonArray;
  typedef std::initializer_list<jsonKeyValue> jsonInitializerList;

  // TODO(v2.0): Remove occa::properties
  typedef json properties;

  typedef struct {
    primitive number;
    std::string string;
    jsonArray array;
    jsonObject object;
  } jsonValue_t;

  class json {
  public:
    static const char objectKeyEndChars[];

    enum type_t {
      none_    = (1 << 0),
      null_    = (1 << 1),
      number_  = (1 << 2),
      string_  = (1 << 3),
      array_   = (1 << 4),
      object_  = (1 << 5)
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
      type(number_) {
      value_.number = value;
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

    inline json(const float value) :
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

    inline json(const char *value) :
      type(string_) {
      value_.string = value;
    }

    inline json(const std::string &value) :
      type(string_) {
      value_.string = value;
    }

    inline json(const hash_t &value) :
      type(string_) {
      value_.string = value.getFullString();
    }

    inline json(const jsonObject &value) :
      type(object_) {
      value_.object = value;
    }

    inline json(const jsonArray &value) :
      type(array_) {
      value_.array = value;
    }

    json(const std::string &name,
         const primitive &value);

    json(std::initializer_list<jsonKeyValue> entries);

    virtual ~json();

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
      type = number_;
      value_.number = value;
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

    inline json& operator = (const float value) {
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

    inline json& operator = (const hash_t &value) {
      type = string_;
      value_.string = value.getFullString();
      return *this;
    }

    inline json& operator = (const jsonObject &value) {
      type = object_;
      value_.object = value;
      return *this;
    }

    inline json& operator = (const jsonArray &value) {
      type = array_;
      value_.array = value;
      return *this;
    }

    virtual bool isInitialized() const;

    json& load(const char *&c);
    json& load(const std::string &s);

    std::string dump(const int indent = 2) const;

    void dumpToString(std::string &out,
                      const std::string &indent = "",
                      const std::string &currentIndent = "") const;

    static json parse(const char *&c);
    static json parse(const std::string &s);

    static json read(const std::string &filename);
    void write(const std::string &filename);

    void loadString(const char *&c);
    void loadNumber(const char *&c);
    void loadObject(const char *&c);
    void loadObjectField(const char *&c);
    void loadArray(const char *&c);
    void loadTrue(const char *&c);
    void loadFalse(const char *&c);
    void loadNull(const char *&c);
    void loadComment(const char *&c);

    json operator + (const json &j) const;
    json& operator += (const json &j);

    void mergeWithObject(const jsonObject &obj);

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

    inline bool isBool() const {
      return (
        (type == number_) &&
        value_.number.isBool()
      );
    }

    inline bool isNull() const {
      return (type == null_);
    }

    inline dtype_t dtype() const {
      switch (type) {
        case null_:
          return dtype::byte;
        case number_:
          return value_.number.dtype();
        default:
          return dtype::none;
      }
    }

    inline json& asNull() {
      if (type & ~(none_ | null_)) {
        clear();
      }
      type = null_;
      return *this;
    }

    inline json& asBoolean() {
      if (type & number_) {
        value_.number = (bool) value_.number;
      } else {
        clear();
        type = number_;
        value_.number = false;
      }
      return *this;
    }

    inline json& asNumber() {
      if (type & ~(none_ | number_)) {
        clear();
      }
      type = number_;
      return *this;
    }

    inline json& asString() {
      if (type & ~(none_ | string_)) {
        clear();
      }
      type = string_;
      return *this;
    }

    inline json& asArray() {
      if (type & ~(none_ | array_)) {
        clear();
      }
      type = array_;
      return *this;
    }

    inline json& asObject() {
      if (type & ~(none_ | object_)) {
        clear();
      }
      type = object_;
      return *this;
    }

    inline bool& boolean() {
      return value_.number.value.bool_;
    }

    inline primitive& number() {
      return value_.number;
    }

    inline std::string& string() {
      return value_.string;
    }

    inline jsonArray& array() {
      return value_.array;
    }

    inline jsonObject& object() {
      return value_.object;
    }

    inline bool boolean() const {
      return value_.number;
    }

    inline const primitive& number() const {
      return value_.number;
    }

    inline const std::string& string() const {
      return value_.string;
    }

    inline const jsonArray& array() const {
      return value_.array;
    }

    inline const jsonObject& object() const {
      return value_.object;
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
    json& set(const char *key,
              const TM &value);

    template <class TM>
    json& set(const std::string &key,
              const TM &value);

    json getPathValue(const char *key) const;

    template <class TM>
    TM get(const char *key,
           const TM &default_ = TM()) const;

    template <class TM>
    TM get(const std::string &key,
           const TM &default_ = TM()) const;

    template <class TM>
    std::vector<TM> toVector(const std::vector<TM> &default_ = std::vector<TM>()) const;

    template <class TM>
    std::vector<TM> toVector(const char *c,
                             const std::vector<TM> &default_ = std::vector<TM>()) const;

    template <class TM>
    std::vector<TM> toVector(const std::string &s,
                             const std::vector<TM> &default_ = std::vector<TM>()) const;

    strVector keys() const;
    jsonArray values() const;

    json& remove(const char *c);

    inline json& remove(const std::string &s) {
      remove(s.c_str());
      return *this;
    }

    inline bool operator == (const json &j) const {
      if (type != j.type) {
        return false;
      }
      switch (type) {
      case none_:
        return true;
      case null_:
        return true;
      case number_:
        return primitive::equal(value_.number, j.value_.number);
      case string_:
        return value_.string == j.value_.string;
      case object_:
        return value_.object == j.value_.object;
      case array_:
        return value_.array == j.value_.array;
      default:
        return false;
      }
    }

    inline operator bool () const {
      switch (type) {
      case number_:
        return value_.number;
      case string_:
        return value_.string.size();
      case object_:
        return true;
      case array_:
        return true;
      default:
        return false;
      }
    }

    inline operator uint8_t () const {
      if (type & number_) {
        return (uint8_t) value_.number;
      } else {
        return 0;
      }
    }

    inline operator uint16_t () const {
      if (type & number_) {
        return (uint16_t) value_.number;
      } else {
        return 0;
      }
    }

    inline operator uint32_t () const {
      if (type & number_) {
        return (uint32_t) value_.number;
      } else {
        return 0;
      }
    }

    inline operator uint64_t () const {
      if (type & number_) {
        return (uint64_t) value_.number;
      } else {
        return 0;
      }
    }

    inline operator int8_t () const {
      if (type & number_) {
        return (int8_t) value_.number;
      } else {
        return 0;
      }
    }

    inline operator int16_t () const {
      if (type & number_) {
        return (int16_t) value_.number;
      } else {
        return 0;
      }
    }

    inline operator int32_t () const {
      if (type & number_) {
        return (int32_t) value_.number;
      } else {
        return 0;
      }
    }

    inline operator int64_t () const {
      if (type & number_) {
        return (int64_t) value_.number;
      } else {
        return 0;
      }
    }

    inline operator float () const {
      if (type & number_) {
        return (float) value_.number;
      } else {
        return 0;
      }
    }

    inline operator double () const {
      if (type & number_) {
        return (double) value_.number;
      } else {
        return 0;
      }
    }

    inline operator std::string () const {
      return toString();
    }

    hash_t hash() const;

    std::string toString() const;

    friend std::ostream& operator << (std::ostream &out,
                                      const json &j);
  };

  class jsonKeyValue {
   public:
    std::string name;
    json value;

    jsonKeyValue(const std::string &name_,
                 const bool value_);

    jsonKeyValue(const std::string &name_,
                 const uint8_t value_);

    jsonKeyValue(const std::string &name_,
                 const int8_t value_);

    jsonKeyValue(const std::string &name_,
                 const uint16_t value_);

    jsonKeyValue(const std::string &name_,
                 const int16_t value_);

    jsonKeyValue(const std::string &name_,
                 const uint32_t value_);

    jsonKeyValue(const std::string &name_,
                 const int32_t value_);

    jsonKeyValue(const std::string &name_,
                 const uint64_t value_);

    jsonKeyValue(const std::string &name_,
                 const int64_t value_);

    jsonKeyValue(const std::string &name_,
                 const float value_);

    jsonKeyValue(const std::string &name_,
                 const double value_);

    jsonKeyValue(const std::string &name_,
                 const primitive &value_);

    jsonKeyValue(const std::string &name_,
                 const char *value_);

    jsonKeyValue(const std::string &name_,
                 const std::string &value_);

    jsonKeyValue(const std::string &name_,
                 const hash_t &value_);

    jsonKeyValue(const std::string &name_,
                 const json &value_);

    jsonKeyValue(const std::string &name_,
                 std::initializer_list<jsonKeyValue> value_);
  };

  template <>
  hash_t hash(const occa::json &json);

  std::ostream& operator << (std::ostream &out,
                           const json &j);
}

#include "json.tpp"

#endif
