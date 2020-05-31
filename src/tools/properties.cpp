#include <occa/io/utils.hpp>
#include <occa/tools/properties.hpp>
#include <occa/tools/string.hpp>

namespace occa {
  properties::properties() {
    type = object_;
    initialized = false;
  }

  properties::properties(const properties &other) : json() {
    type = object_;
    value_ = other.value_;

    // Note: "other" might be a json object
    initialized = other.isInitialized();
  }

  properties::properties(const json &j) {
    type = object_;
    value_ = j.value_;
    initialized = true;
  }

  properties::properties(const char *c) :
      initialized(false) {
    properties::load(c);
  }

  properties::properties(const std::string &s) :
      initialized(false) {
    properties::load(s);
  }

  properties::~properties() {}

  bool properties::isInitialized() const {
    if (!initialized) {
      initialized = value_.object.size();
    }
    return initialized;
  }

  void properties::load(const char *&c) {
    lex::skipWhitespace(c);
    loadObject(c);
    initialized = true;
  }

  void properties::load(const std::string &s) {
    const char *c = s.c_str();
    lex::skipWhitespace(c);
    loadObject(c);
    initialized = true;
  }

  properties properties::read(const std::string &filename) {
    properties props;
    props.load(io::read(filename));
    return props;
  }

  template <>
  hash_t hash(const properties &props) {
    return props.hash();
  }
}
