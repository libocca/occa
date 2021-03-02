#ifndef OCCA_FUNCTIONAL_TYPES_HEADER
#define OCCA_FUNCTIONAL_TYPES_HEADER

namespace occa {
  enum class reductionType {
    sum,
    multiply,
    bitOr,
    bitAnd,
    bitXor,
    boolOr,
    boolAnd,
    min,
    max
  };
}

#endif
