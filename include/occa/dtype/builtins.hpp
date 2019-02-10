#ifndef OCCA_DTYPE_BUILTINS_HEADER
#define OCCA_DTYPE_BUILTINS_HEADER

#include <occa/dtype/dtype.hpp>

namespace occa {
  namespace dtype {
    extern const dtype_t none;

    extern const dtype_t void_;
    extern const dtype_t byte;

    extern const dtype_t bool_;
    extern const dtype_t char_;
    extern const dtype_t short_;
    extern const dtype_t int_;
    extern const dtype_t long_;
    extern const dtype_t float_;
    extern const dtype_t double_;

    extern const dtype_t int8;
    extern const dtype_t uint8;
    extern const dtype_t int16;
    extern const dtype_t uint16;
    extern const dtype_t int32;
    extern const dtype_t uint32;
    extern const dtype_t int64;
    extern const dtype_t uint64;
    extern const dtype_t float32;
    extern const dtype_t float64;

    // OKL Primitives
    extern const dtype_t uchar2;
    extern const dtype_t uchar3;
    extern const dtype_t uchar4;

    extern const dtype_t char2;
    extern const dtype_t char3;
    extern const dtype_t char4;

    extern const dtype_t ushort2;
    extern const dtype_t ushort3;
    extern const dtype_t ushort4;

    extern const dtype_t short2;
    extern const dtype_t short3;
    extern const dtype_t short4;

    extern const dtype_t uint2;
    extern const dtype_t uint3;
    extern const dtype_t uint4;

    extern const dtype_t int2;
    extern const dtype_t int3;
    extern const dtype_t int4;

    extern const dtype_t ulong2;
    extern const dtype_t ulong3;
    extern const dtype_t ulong4;

    extern const dtype_t long2;
    extern const dtype_t long3;
    extern const dtype_t long4;

    extern const dtype_t float2;
    extern const dtype_t float3;
    extern const dtype_t float4;

    extern const dtype_t double2;
    extern const dtype_t double3;
    extern const dtype_t double4;
  }
}

#endif
