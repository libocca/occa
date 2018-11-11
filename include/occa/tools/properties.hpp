#ifndef OCCA_TOOLS_PROPERTIES_HEADER
#define OCCA_TOOLS_PROPERTIES_HEADER

#include <occa/tools/json.hpp>

namespace occa {
  class properties: public json {
  public:
    bool initialized;

    properties();
    properties(const properties &other);
    properties(const json &j);
    properties(const char *c);
    properties(const std::string &s);
    ~properties();

    bool isInitialized();

    void load(const char *&c);
    void load(const std::string &s);

    static properties read(const std::string &filename);

    inline properties operator + (const properties &p) const {
      properties ret = *this;
      ret.mergeWithObject(p.value_.object);
      return ret;
    }
  };

  template <>
  hash_t hash(const properties &props);
}

#endif
