#ifndef OCCA_INTERNAL_LANG_BUILTINS_TYPES_HEADER
#define OCCA_INTERNAL_LANG_BUILTINS_TYPES_HEADER

#include <occa/internal/lang/type.hpp>

namespace occa {
  namespace lang {
    extern const qualifier_t const_;
    extern const qualifier_t constexpr_;
    extern const qualifier_t friend_;
    extern const qualifier_t typedef_;
    extern const qualifier_t signed_;
    extern const qualifier_t unsigned_;
    extern const qualifier_t volatile_;
    extern const qualifier_t long_;
    extern const qualifier_t longlong_;

    extern const qualifier_t extern_;
    extern const qualifier_t externC;
    extern const qualifier_t externCpp;
    extern const qualifier_t mutable_;
    extern const qualifier_t register_;
    extern const qualifier_t static_;
    extern const qualifier_t thread_local_;

    extern const qualifier_t explicit_;
    extern const qualifier_t inline_;
    extern const qualifier_t virtual_;

    extern const qualifier_t class_;
    extern const qualifier_t struct_;
    extern const qualifier_t enum_;
    extern const qualifier_t union_;

    // Windows types
    // TODO: Properly handle compiler extension attributes
    extern const qualifier_t dllexport_;

    extern const primitive_t bool_;
    extern const primitive_t char_;
    extern const primitive_t char16_t_;
    extern const primitive_t char32_t_;
    extern const primitive_t wchar_t_;
    extern const primitive_t short_;
    extern const primitive_t int_;
    extern const primitive_t float_;
    extern const primitive_t double_;
    extern const primitive_t void_;
    extern const primitive_t auto_;

    // TODO: Auto-generate type aliases for common C types
    extern const primitive_t size_t_;
    extern const primitive_t ptrdiff_t_;

    // OKL Primitives
    extern const primitive_t uchar2;
    extern const primitive_t uchar3;
    extern const primitive_t uchar4;

    extern const primitive_t char2;
    extern const primitive_t char3;
    extern const primitive_t char4;

    extern const primitive_t ushort2;
    extern const primitive_t ushort3;
    extern const primitive_t ushort4;

    extern const primitive_t short2;
    extern const primitive_t short3;
    extern const primitive_t short4;

    extern const primitive_t uint2;
    extern const primitive_t uint3;
    extern const primitive_t uint4;

    extern const primitive_t int2;
    extern const primitive_t int3;
    extern const primitive_t int4;

    extern const primitive_t ulong2;
    extern const primitive_t ulong3;
    extern const primitive_t ulong4;

    extern const primitive_t long2;
    extern const primitive_t long3;
    extern const primitive_t long4;

    extern const primitive_t float2;
    extern const primitive_t float3;
    extern const primitive_t float4;

    extern const primitive_t double2;
    extern const primitive_t double3;
    extern const primitive_t double4;
    
    // DPCPP Primitives
    extern const primitive_t syclQueue;
    extern const primitive_t syclNdRange;
    extern const primitive_t syclNdItem;
    extern const primitive_t syclHandler;
    extern const primitive_t syclAccessor;
  } // namespace lang
}
#endif
