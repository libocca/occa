#ifndef OCCA_TOOLS_PROPERTIES_HEADER
#define OCCA_TOOLS_PROPERTIES_HEADER

#include <occa/tools/json.hpp>

namespace occa {
  class properties: public json {
  public:
    // Note: Do not use directly since we're casting between
    //       occa::json& <--> occa::properties&
    mutable bool initialized;

    properties();
    properties(const properties &other);
    properties(const json &j);
    properties(const char *c);
    properties(const std::string &s);
    ~properties();

    bool isInitialized() const;

    void load(const char *&c);
    void load(const std::string &s);

    static properties read(const std::string &filename);
  };

  inline properties operator + (const properties &left, const properties &right) {
    properties sum = left;
    sum.mergeWithObject(right.value_.object);
    return sum;
  }

  template <>
  hash_t hash(const properties &props);
}

#endif
