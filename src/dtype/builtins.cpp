#include <occa/dtype/builtins.hpp>

namespace occa {
  namespace dtype {
    const dtype_t none("none", 0, true);

    const dtype_t void_("void", 0, true);
    const dtype_t byte("byte", 1, true);

    const dtype_t bool_("bool", sizeof(bool), true);
    const dtype_t char_("char", sizeof(char), true);
    const dtype_t short_("short", sizeof(short), true);
    const dtype_t int_("int", sizeof(int), true);
    const dtype_t long_("long", sizeof(long), true);
    const dtype_t float_("float", sizeof(float), true);
    const dtype_t double_("double", sizeof(double), true);

    const dtype_t int8("int8", 1, true);
    const dtype_t uint8("uint8", 1, true);
    const dtype_t int16("int16", 2, true);
    const dtype_t uint16("uint16", 2, true);
    const dtype_t int32("int32", 4, true);
    const dtype_t uint32("uint32", 4, true);
    const dtype_t int64("int64", 8, true);
    const dtype_t uint64("uint64", 8, true);
    const dtype_t float32("float32", 4, true);
    const dtype_t float64("float64", 8, true);

    // OKL Primitives
    const dtype_t uchar2("uchar2", 2 * sizeof(char), true);
    const dtype_t uchar3("uchar3", 3 * sizeof(char), true);
    const dtype_t uchar4("uchar4", 4 * sizeof(char), true);

    const dtype_t char2("char2", 2 * sizeof(char), true);
    const dtype_t char3("char3", 3 * sizeof(char), true);
    const dtype_t char4("char4", 4 * sizeof(char), true);

    const dtype_t ushort2("ushort2", 2 * sizeof(short), true);
    const dtype_t ushort3("ushort3", 3 * sizeof(short), true);
    const dtype_t ushort4("ushort4", 4 * sizeof(short), true);

    const dtype_t short2("short2", 2 * sizeof(short), true);
    const dtype_t short3("short3", 3 * sizeof(short), true);
    const dtype_t short4("short4", 4 * sizeof(short), true);

    const dtype_t uint2("uint2", 2 * sizeof(int), true);
    const dtype_t uint3("uint3", 3 * sizeof(int), true);
    const dtype_t uint4("uint4", 4 * sizeof(int), true);

    const dtype_t int2("int2", 2 * sizeof(int), true);
    const dtype_t int3("int3", 3 * sizeof(int), true);
    const dtype_t int4("int4", 4 * sizeof(int), true);

    const dtype_t ulong2("ulong2", 2 * sizeof(long), true);
    const dtype_t ulong3("ulong3", 3 * sizeof(long), true);
    const dtype_t ulong4("ulong4", 4 * sizeof(long), true);

    const dtype_t long2("long2", 2 * sizeof(long), true);
    const dtype_t long3("long3", 3 * sizeof(long), true);
    const dtype_t long4("long4", 4 * sizeof(long), true);

    const dtype_t float2("float2", 2 * sizeof(float), true);
    const dtype_t float3("float3", 3 * sizeof(float), true);
    const dtype_t float4("float4", 4 * sizeof(float), true);

    const dtype_t double2("double2", 2 * sizeof(double), true);
    const dtype_t double3("double3", 3 * sizeof(double), true);
    const dtype_t double4("double4", 4 * sizeof(double), true);
  }
}
