#ifndef OCCA_UTILS_KEYVALUE_HEADER
#define OCCA_UTILS_KEYVALUE_HEADER

#include <iostream>

namespace occa {
  template <class TM>
  class keyValue_t {
   public:
    const std::string key;
    const TM value;

    inline keyValue_t(const std::string &key_,
                      const TM &value_) :
        key(key_),
        value(value_) {}
  };
}

#endif
