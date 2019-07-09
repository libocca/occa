#ifndef OCCA_LANG_BUILTINS_ATTRIBUTES_TILE_HEADER
#define OCCA_LANG_BUILTINS_ATTRIBUTES_TILE_HEADER

#include <occa/lang/attribute.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      class tile : public attribute_t {
      public:
        tile();

        virtual const std::string& name() const;

        virtual bool forStatement(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;
        bool validArgs(const attributeToken_t &attr) const;
        bool validKwargs(const attributeToken_t &attr) const;
      };
    }
  }
}

#endif
