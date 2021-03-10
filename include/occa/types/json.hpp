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

  /**
   * @startDoc{json}
   *
   * Description:
   *
   *   A [[json]] object stores data in the same way specified by the JSON standard.
   *   It's used across the OCCA library as a way to flexibly pass user configurations.
   *
   *   # Types
   *
   *   There are 6 basic types a json object can be at a time:
   *   - String
   *   - Number
   *   - Boolean
   *   - NULL
   *   - Array of json objects
   *   - Map of string keys to json objects
   *
   *   # Type checking
   *
   *   There is a method provided check for each type
   *
   *   - [[json.isString]]
   *   - [[json.isNumber]]
   *   - [[json.isBool]]
   *   - [[json.isNull]]
   *   - [[json.isObject]]
   *   - [[json.isArray]]
   *
   *   # Type casting
   *
   *    There is also a method to enforce the json object to be a specific type
   *
   *   - [[json.asString]]
   *   - [[json.asNumber]]
   *   - [[json.asBoolean]]
   *   - [[json.asNull]]
   *   - [[json.asObject]]
   *   - [[json.asArray]]
   *
   *   # Data access
   *
   *   Accessing and setting data can be done through the [[json.operator_brackets]].
   *   To make it simpler to access nested structures, we support passing `/`-delimited paths
   *
   *   For example, if we wanted to build
   *
   *   ```js
   *   {
   *     "a": {
   *       "b": {
   *         "c": "hello world"
   *       }
   *     }
   *   }
   *   ```
   *
   *   we could do it two ways:
   *
   *   ```cpp
   *   occa::json j;
   *   j["a"]["b"]["c"] = "hello world';
   *   ```
   *
   *   or a the more compact way:
   *
   *   ```cpp
   *   occa::json j;
   *   j["a/b/c"] = "hello world';
   *   ```
   *
   *   If for some reason there needs to be a `/` in the key name, use the [[json.set]] method instead
   *
   *   For example, building
   *
   *   ```js
   *   {
   *     "a/b/c": "hello world"
   *   }
   *   ```
   *
   *   would be done through
   *
   *   ```cpp
   *   occa::json j;
   *   j.set("a/b/c", "hello world');
   *   ```
   *
   *   # Decoding
   *
   *   - [[json.parse]] can be used to parse a string to a json object.
   *   - [[json.read]] is the same as [[json.parse]] but reads and parses a file instead.
   *
   *   # Encoding
   *   - [[json.dump]] produces the JSON string associated with the stored data.
   *   - [[json.write]] is the same as [[json.dump]] but saves the output in a file.
   *
   * @endDoc
   */
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

    /**
     * @startDoc{dump}
     *
     * Description:
     *   Return the JSON-formatted string representation of the data
     *
     *   Same as [[json.write]] but to a `string` rather than a file
     *
     * @endDoc
     */
    std::string dump(const int indent = 2) const;

    void dumpToString(std::string &out,
                      const std::string &indent = "",
                      const std::string &currentIndent = "") const;

    /**
     * @startDoc{parse[0]}
     *
     * Description:
     *   Parse a JSON-formatted string.
     *   Throw an `occa::exception` if the input is not of JSON-format
     *
     *   Same as [[json.read]] but with a file rather than a `string`.
     *
     * @endDoc
     */
    static json parse(const char *&c);

    /**
     * @doc{parse[1]}
     */
    static json parse(const std::string &s);

    /**
     * @startDoc{read}
     *
     * Description:
     *   Read the input file and parse the JSON-formatted contents.
     *   Throw an `occa::exception` if the input is not of JSON-format
     *
     *   Same as [[json.parse]] but with a file rather than a `string`.
     *
     * @endDoc
     */
    static json read(const std::string &filename);

    /**
     * @startDoc{write}
     *
     * Description:
     *   Output the JSON-formatted string representation of the data into the given file.
     *
     *   Same as [[json.dump]] but writes to a file rather than a `string`.
     *
     * @endDoc
     */
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

    /**
     * @startDoc{has}
     *
     * Description:
     *   If it's an object, return whether it has a key `s` which handles paths.
     *
     *   For example, the following checks whether there is a nested `{a: {b: {c: ...}}}` structure:
     *
     *   ```cpp
     *   j.has("a/b/c")
     *   ```
     *
     * @endDoc
     */
    bool has(const std::string &key) const;

    /**
     * @startDoc{isString}
     *
     * Description:
     *   Returns whether the json object represents a `string`
     *
     * @endDoc
     */
    inline bool isString() const {
      return (type == string_);
    }

    /**
     * @startDoc{isNumber}
     *
     * Description:
     *   Returns whether the json object represents a `number`
     *
     * @endDoc
     */
    inline bool isNumber() const {
      return (type == number_);
    }

    /**
     * @startDoc{isObject}
     *
     * Description:
     *   Returns whether the json object represents an `object`
     *
     * @endDoc
     */
    inline bool isObject() const {
      return (type == object_);
    }

    /**
     * @startDoc{isArray}
     *
     * Description:
     *   Returns whether the json object represents an `array`
     *
     * @endDoc
     */
    inline bool isArray() const {
      return (type == array_);
    }

    /**
     * @startDoc{isBool}
     *
     * Description:
     *   Returns whether the json object represents an `bool`
     *
     * @endDoc
     */
    inline bool isBool() const {
      return (
        (type == number_) &&
        value_.number.isBool()
      );
    }

    /**
     * @startDoc{isNull}
     *
     * Description:
     *   Returns whether the json object represents `NULL`
     *
     * @endDoc
     */
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

    /**
     * @startDoc{asNull}
     *
     * Description:
     *   Cast the JSON type as `NULL`
     *
     * @endDoc
     */
    inline json& asNull() {
      if (type & ~(none_ | null_)) {
        clear();
      }
      type = null_;
      return *this;
    }

    /**
     * @startDoc{asBoolean}
     *
     * Description:
     *   Cast the JSON type as `boolean`
     *
     * @endDoc
     */
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

    /**
     * @startDoc{asNumber}
     *
     * Description:
     *   Cast the JSON type as a `number`
     *
     * @endDoc
     */
    inline json& asNumber() {
      if (type & ~(none_ | number_)) {
        clear();
      }
      type = number_;
      return *this;
    }

    /**
     * @startDoc{asString}
     *
     * Description:
     *   Cast the JSON type as a `stream`
     *
     * @endDoc
     */
    inline json& asString() {
      if (type & ~(none_ | string_)) {
        clear();
      }
      type = string_;
      return *this;
    }

    /**
     * @startDoc{asArray}
     *
     * Description:
     *   Cast the JSON type as an `array`
     *
     * @endDoc
     */
    inline json& asArray() {
      if (type & ~(none_ | array_)) {
        clear();
      }
      type = array_;
      return *this;
    }

    /**
     * @startDoc{asObject}
     *
     * Description:
     *   Cast the JSON type as an `object`
     *
     * @endDoc
     */
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

    /**
     * @startDoc{operator_brackets[0]}
     *
     * Description:
     *   Access data if it's an object or array.
     *
     * Overloaded Description:
     *   Return the object value given by key `s`
     *
     * @endDoc
     */
    inline json& operator [] (const std::string &s) {
      return (*this)[s.c_str()];
    }

    /**
     * @doc{operator_brackets[1]}
     */
    inline const json& operator [] (const std::string &s) const {
      return (*this)[s.c_str()];
    }

    /**
     * @startDoc{operator_brackets[2]}
     *
     * Overloaded Description:
     *   Return the value at array on index `n`
     *
     * @endDoc
     */
    json& operator [] (const int n);

    /**
     * @doc{operator_brackets[3]}
     */
    const json& operator [] (const int n) const;

    /**
     * @startDoc{size}
     *
     * Description:
     *   If it's an object, return how many key/value pairs it has.
     *
     *   If it's an array, return how many array entries it has.
     *
     *   If it's a string, return its length.
     *
     *   Otherwise, return 0
     */
    int size() const;

    /**
     * @startDoc{set[0]}
     *
     * Description:
     *   Similar to [[json.operator_brackets]] which can be used to set a value, but avoids parsing `/` as paths
     *
     * @endDoc
     */
    template <class T>
    json& set(const char *key,
              const T &value);

    /**
     * @doc{set[1]}
     */
    template <class T>
    json& set(const std::string &key,
              const T &value);

    json getPathValue(const char *key) const;

    /**
     * @startDoc{get[0]}
     *
     * Description:
     *   Similar to [[json.operator_brackets]] which can be used to get a value, but avoids parsing `/` as paths
     *
     * @endDoc
     */
    template <class T>
    T get(const char *key,
           const T &default_ = T()) const;

    /**
     * @doc{get[1]}
     */
    template <class T>
    T get(const std::string &key,
           const T &default_ = T()) const;

    template <class T>
    std::vector<T> toVector(const std::vector<T> &default_ = std::vector<T>()) const;

    template <class T>
    std::vector<T> toVector(const char *c,
                             const std::vector<T> &default_ = std::vector<T>()) const;

    template <class T>
    std::vector<T> toVector(const std::string &s,
                             const std::vector<T> &default_ = std::vector<T>()) const;

    /**
     * @startDoc{keys}
     *
     * Description:
     *   If it's an object, return an `std::vector<string>` of its keys.
     *
     *   Otherwise return an empty vector.
     *
     * @endDoc
     */
    strVector keys() const;

    /**
     * @startDoc{value}
     *
     * Description:
     *   If it's an object, return an `std::vector<string>` of its values.
     *
     *   Otherwise return an empty vector.
     *
     * @endDoc
     */
    jsonArray values() const;

    /**
     * @startDoc{remove[0]}
     *
     * Description:
     *   If it's an object, remove the key/value corresponding which matches the key
     *
     * @endDoc
     */
    json& remove(const char *c);

    /**
     * @doc{remove[1]}
     */
    inline json& remove(const std::string &s) {
      remove(s.c_str());
      return *this;
    }

    /**
     * @startDoc{operator_equals}
     *
     * Description:
     *   Returns `true` if the two json values are the same
     *
     * @endDoc
     */
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
