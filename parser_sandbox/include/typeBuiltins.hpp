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
    extern const qualifier enum_;
    extern const qualifier struct_;
    extern const qualifier union_;

    extern const primitive bool_;
    extern const primitive char_;
    extern const primitive char16_t_;
    extern const primitive char32_t_;
    extern const primitive wchar_t_;
    extern const primitive short_;
    extern const primitive int_;
    extern const primitive long_;
    extern const primitive float_;
    extern const primitive double_;
    extern const primitive void_;
    extern const primitive auto_;
  }
}
#endif
