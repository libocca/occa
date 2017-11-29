#ifndef OCCA_PARSER_TYPEBUILTINS_HEADER2
#define OCCA_PARSER_TYPEBUILTINS_HEADER2

#include "type.hpp"

namespace occa {
  namespace lang {
    extern const qualifier const_;
    extern const qualifier constexpr_;
    extern const qualifier friend_;
    extern const qualifier typedef_;
    extern const qualifier signed_;
    extern const qualifier unsigned_;
    extern const qualifier volatile_;

    extern const qualifier extern_;
    extern const qualifier mutable_;
    extern const qualifier register_;
    extern const qualifier static_;
    extern const qualifier thread_local_;

    extern const qualifier explicit_;
    extern const qualifier inline_;
    extern const qualifier virtual_;

    extern const qualifier class_;
    extern const qualifier struct_;
    extern const qualifier enum_;
    extern const qualifier union_;

    extern const primitiveType bool_;
    extern const primitiveType char_;
    extern const primitiveType char16_t_;
    extern const primitiveType char32_t_;
    extern const primitiveType wchar_t_;
    extern const primitiveType short_;
    extern const primitiveType int_;
    extern const primitiveType long_;
    extern const primitiveType float_;
    extern const primitiveType double_;
    extern const primitiveType void_;
    extern const primitiveType auto_;
  }
}
#endif
