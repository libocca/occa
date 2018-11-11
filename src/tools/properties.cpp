#include <occa/io/utils.hpp>
#include <occa/tools/properties.hpp>
#include <occa/tools/string.hpp>

namespace occa {
  properties::properties() {
    type = object_;
    initialized = false;
  }

  properties::properties(const properties &other) {
    type = object_;
    value_ = other.value_;
    initialized = other.initialized;
  }

  properties::properties(const json &j) {
    type = object_;
    value_ = j.value_;
    initialized = true;
  }

  properties::properties(const char *c) {
    properties::load(c);
  }

  properties::properties(const std::string &s) {
    properties::load(s);
  }

  properties::~properties() {}

  bool properties::isInitialized() {
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
