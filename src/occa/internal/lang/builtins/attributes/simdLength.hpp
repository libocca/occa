#ifndef OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_SIMD_LENGTH_HEADER
#define OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_SIMD_LENGTH_HEADER

#include <occa/internal/lang/attribute.hpp>

namespace occa {
namespace lang {
namespace attributes {

class simdLength : public attribute_t {
public:
  simdLength() = default;
  const std::string& name() const override;
  bool forStatementType(const int sType) const override;
  bool isValid(const attributeToken_t &attr) const override;
private:
  static const inline std::string name_{"simd_length"};
};

}
}
}

#endif
