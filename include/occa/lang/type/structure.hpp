#ifndef OCCA_LANG_TYPE_STRUCTURE_HEADER
#define OCCA_LANG_TYPE_STRUCTURE_HEADER

#include <occa/lang/type/type.hpp>

namespace occa {
  namespace lang {
    class structure_t : public type_t {
    public:
      blockStatement *body;

      structure_t(const std::string &name_ = "");
    };
  }
}

#endif
