#include "typeBuiltins.hpp"

namespace occa {
  namespace lang {
    const qualifier const_     ("const");
    const qualifier constexpr_ ("constexpr");
    const qualifier friend_    ("friend");
    const qualifier typedef_   ("typedef");
    const qualifier signed_    ("signed");
    const qualifier unsigned_  ("unsigned");
    const qualifier volatile_  ("volatile");

    const qualifier extern_       ("extern"      , specifier::storageType);
    const qualifier mutable_      ("mutable"     , specifier::storageType);
    const qualifier register_     ("register"    , specifier::storageType);
    const qualifier static_       ("static"      , specifier::storageType);
    const qualifier thread_local_ ("thread_local", specifier::storageType);

    const qualifier explicit_ ("explicit", specifier::functionType);
    const qualifier inline_   ("inline"  , specifier::functionType);
    const qualifier virtual_  ("virtual" , specifier::functionType);

    const qualifier class_  ("class" , specifier::variableType);
    const qualifier enum_   ("enum"  , specifier::variableType);
    const qualifier struct_ ("struct", specifier::variableType);
    const qualifier union_  ("union" , specifier::variableType);

    const primitive bool_     ("bool");
    const primitive char_     ("char");
    const primitive char16_t_ ("char16_t");
    const primitive char32_t_ ("char32_t");
    const primitive wchar_t_  ("wchar_t");
    const primitive short_    ("short");
    const primitive int_      ("int");
    const primitive long_     ("long");
    const primitive float_    ("float");
    const primitive double_   ("double");
    const primitive void_     ("void");
    const primitive auto_     ("auto");
  }
}
