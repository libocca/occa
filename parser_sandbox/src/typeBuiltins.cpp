#include "typeBuiltins.hpp"

namespace occa {
  namespace lang {
    const qualifier const_        ("const"       , qualifierType::const_);
    const qualifier constexpr_    ("constexpr"   , qualifierType::constexpr_);
    const qualifier friend_       ("friend"      , qualifierType::friend_);
    const qualifier typedef_      ("typedef"     , qualifierType::typedef_);
    const qualifier signed_       ("signed"      , qualifierType::signed_);
    const qualifier unsigned_     ("unsigned"    , qualifierType::unsigned_);
    const qualifier volatile_     ("volatile"    , qualifierType::volatile_);

    const qualifier extern_       ("extern"      , qualifierType::extern_);
    const qualifier mutable_      ("mutable"     , qualifierType::mutable_);
    const qualifier register_     ("register"    , qualifierType::register_);
    const qualifier static_       ("static"      , qualifierType::static_);
    const qualifier thread_local_ ("thread_local", qualifierType::thread_local_);

    const qualifier explicit_     ("explicit"    , qualifierType::explicit_);
    const qualifier inline_       ("inline"      , qualifierType::inline_);
    const qualifier virtual_      ("virtual"     , qualifierType::virtual_);

    const qualifier class_        ("class"       , qualifierType::class_);
    const qualifier enum_         ("enum"        , qualifierType::enum_);
    const qualifier struct_       ("struct"      , qualifierType::struct_);
    const qualifier union_        ("union"       , qualifierType::union_);

    const primitiveType bool_     ("bool"    );
    const primitiveType char_     ("char"    );
    const primitiveType char16_t_ ("char16_t");
    const primitiveType char32_t_ ("char32_t");
    const primitiveType wchar_t_  ("wchar_t" );
    const primitiveType short_    ("short"   );
    const primitiveType int_      ("int"     );
    const primitiveType long_     ("long"    );
    const primitiveType float_    ("float"   );
    const primitiveType double_   ("double"  );
    const primitiveType void_     ("void"    );
    const primitiveType auto_     ("auto"    );
  }
}
