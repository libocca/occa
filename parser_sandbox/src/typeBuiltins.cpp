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

    const primitiveType bool_     ("bool");
    const primitiveType char_     ("char");
    const primitiveType char16_t_ ("char16_t");
    const primitiveType char32_t_ ("char32_t");
    const primitiveType wchar_t_  ("wchar_t");
    const primitiveType short_    ("short");
    const primitiveType int_      ("int");
    const primitiveType long_     ("long");
    const primitiveType float_    ("float");
    const primitiveType double_   ("double");
    const primitiveType void_     ("void");
    const primitiveType auto_     ("auto");
  }
}
