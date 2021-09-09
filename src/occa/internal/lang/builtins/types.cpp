#include <occa/internal/lang/builtins/types.hpp>

namespace occa {
  namespace lang {
    const qualifier_t const_        ("const"         , qualifierType::const_);
    const qualifier_t constexpr_    ("constexpr"     , qualifierType::constexpr_);
    const qualifier_t friend_       ("friend"        , qualifierType::friend_);
    const qualifier_t typedef_      ("typedef"       , qualifierType::typedef_);
    const qualifier_t signed_       ("signed"        , qualifierType::signed_);
    const qualifier_t unsigned_     ("unsigned"      , qualifierType::unsigned_);
    const qualifier_t volatile_     ("volatile"      , qualifierType::volatile_);
    const qualifier_t long_         ("long"          , qualifierType::long_);
    const qualifier_t longlong_     ("long long"     , qualifierType::longlong_);

    const qualifier_t extern_       ("extern"        , qualifierType::extern_);
    const qualifier_t externC       ("extern \"C\""  , qualifierType::externC);
    const qualifier_t externCpp     ("extern \"C++\"", qualifierType::externCpp);
    const qualifier_t mutable_      ("mutable"       , qualifierType::mutable_);
    const qualifier_t register_     ("register"      , qualifierType::register_);
    const qualifier_t static_       ("static"        , qualifierType::static_);
    const qualifier_t thread_local_ ("thread_local"  , qualifierType::thread_local_);

    const qualifier_t explicit_     ("explicit"      , qualifierType::explicit_);
    const qualifier_t inline_       ("inline"        , qualifierType::inline_);
    const qualifier_t virtual_      ("virtual"       , qualifierType::virtual_);

    const qualifier_t class_        ("class"         , qualifierType::class_);
    const qualifier_t enum_         ("enum"          , qualifierType::enum_);
    const qualifier_t struct_       ("struct"        , qualifierType::struct_);
    const qualifier_t union_        ("union"         , qualifierType::union_);

    // Windows types
    // TODO: Properly handle compiler extension attributes
    const qualifier_t dllexport_    ("__declspec(dllexport)", qualifierType::dllexport_);

    const primitive_t bool_         ("bool");
    const primitive_t char_         ("char");
    const primitive_t char16_t_     ("char16_t");
    const primitive_t char32_t_     ("char32_t");
    const primitive_t wchar_t_      ("wchar_t");
    const primitive_t short_        ("short");
    const primitive_t int_          ("int");
    const primitive_t float_        ("float");
    const primitive_t double_       ("double");
    const primitive_t void_         ("void");
    const primitive_t auto_         ("auto");

    // TODO: Auto-generate type aliases for common C types
    const primitive_t size_t_       ("size_t");
    const primitive_t ptrdiff_t_    ("ptrdiff_t");

    // OKL Primitives
    const primitive_t uchar2        ("uchar2");
    const primitive_t uchar3        ("uchar3");
    const primitive_t uchar4        ("uchar4");

    const primitive_t char2         ("char2");
    const primitive_t char3         ("char3");
    const primitive_t char4         ("char4");

    const primitive_t ushort2       ("ushort2");
    const primitive_t ushort3       ("ushort3");
    const primitive_t ushort4       ("ushort4");

    const primitive_t short2        ("short2");
    const primitive_t short3        ("short3");
    const primitive_t short4        ("short4");

    const primitive_t uint2         ("uint2");
    const primitive_t uint3         ("uint3");
    const primitive_t uint4         ("uint4");

    const primitive_t int2          ("int2");
    const primitive_t int3          ("int3");
    const primitive_t int4          ("int4");

    const primitive_t ulong2        ("ulong2");
    const primitive_t ulong3        ("ulong3");
    const primitive_t ulong4        ("ulong4");

    const primitive_t long2         ("long2");
    const primitive_t long3         ("long3");
    const primitive_t long4         ("long4");

    const primitive_t float2        ("float2");
    const primitive_t float3        ("float3");
    const primitive_t float4        ("float4");

    const primitive_t double2       ("double2");
    const primitive_t double3       ("double3");
    const primitive_t double4       ("double4");
    
  } // namespace lang
}
