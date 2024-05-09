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
    const dtype_t ulong_("unsigned long", sizeof(unsigned long), true);
    const dtype_t float_("float", sizeof(float), true);
    const dtype_t double_("double", sizeof(double), true);

    const dtype_t int8    = get<int8_t>();
    const dtype_t uint8   = get<uint8_t>();
    const dtype_t int16   = get<int16_t>();
    const dtype_t uint16  = get<uint16_t>();
    const dtype_t int32   = get<int32_t>();
    const dtype_t uint32  = get<uint32_t>();
    const dtype_t int64   = get<int64_t>();
    const dtype_t uint64  = get<uint64_t>();

    // OKL Primitives
    const dtype_t uchar2("uchar2", dtype_t::tuple(char_, 2), true);
    const dtype_t uchar3("uchar3", dtype_t::tuple(char_, 3), true);
    const dtype_t uchar4("uchar4", dtype_t::tuple(char_, 4), true);

    const dtype_t char2("char2", dtype_t::tuple(char_, 2), true);
    const dtype_t char3("char3", dtype_t::tuple(char_, 3), true);
    const dtype_t char4("char4", dtype_t::tuple(char_, 4), true);

    const dtype_t ushort2("ushort2", dtype_t::tuple(short_, 2), true);
    const dtype_t ushort3("ushort3", dtype_t::tuple(short_, 3), true);
    const dtype_t ushort4("ushort4", dtype_t::tuple(short_, 4), true);

    const dtype_t short2("short2", dtype_t::tuple(short_, 2), true);
    const dtype_t short3("short3", dtype_t::tuple(short_, 3), true);
    const dtype_t short4("short4", dtype_t::tuple(short_, 4), true);

    const dtype_t uint2("uint2", dtype_t::tuple(int_, 2), true);
    const dtype_t uint3("uint3", dtype_t::tuple(int_, 3), true);
    const dtype_t uint4("uint4", dtype_t::tuple(int_, 4), true);

    const dtype_t int2("int2", dtype_t::tuple(int_, 2), true);
    const dtype_t int3("int3", dtype_t::tuple(int_, 3), true);
    const dtype_t int4("int4", dtype_t::tuple(int_, 4), true);

    const dtype_t ulong2("ulong2", dtype_t::tuple(long_, 2), true);
    const dtype_t ulong3("ulong3", dtype_t::tuple(long_, 3), true);
    const dtype_t ulong4("ulong4", dtype_t::tuple(long_, 4), true);

    const dtype_t long2("long2", dtype_t::tuple(long_, 2), true);
    const dtype_t long3("long3", dtype_t::tuple(long_, 3), true);
    const dtype_t long4("long4", dtype_t::tuple(long_, 4), true);

    const dtype_t float2("float2", dtype_t::tuple(float_, 2), true);
    const dtype_t float3("float3", dtype_t::tuple(float_, 3), true);
    const dtype_t float4("float4", dtype_t::tuple(float_, 4), true);

    const dtype_t double2("double2", dtype_t::tuple(double_, 2), true);
    const dtype_t double3("double3", dtype_t::tuple(double_, 3), true);
    const dtype_t double4("double4", dtype_t::tuple(double_, 4), true);

    // OCCA Types
    const dtype_t memory("occa::memory", 0, true);

    // Templated types
    template <> dtype_t get<void>() {
      return void_;
    }

    // Primitive types
    template <> dtype_t get<bool>() {
      return bool_;
    }

    template <> dtype_t get<char>() {
      return char_;
    }

    template <> dtype_t get<signed char>() {
      return char_;
    }

    template <> dtype_t get<unsigned char>() {
      return char_;
    }

    template <> dtype_t get<short>() {
      return short_;
    }

    template <> dtype_t get<unsigned short>() {
      return short_;
    }

    template <> dtype_t get<int>() {
      return int_;
    }

    template <> dtype_t get<unsigned int>() {
      return int_;
    }

    template <> dtype_t get<long>() {
      return long_;
    }

    template <> dtype_t get<unsigned long>() {
      return ulong_;
    }

    template <> dtype_t get<long long>() {
      return long_;
    }

    template <> dtype_t get<unsigned long long>() {
      return long_;
    }

    template <> dtype_t get<float>() {
      return float_;
    }

    template <> dtype_t get<double>() {
      return double_;
    }

    template <> dtype_t get<occa::uchar2>() {
      return uchar2;
    }

    template <> dtype_t get<occa::uchar4>() {
      return uchar4;
    }

    template <> dtype_t get<occa::char2>() {
      return char2;
    }

    template <> dtype_t get<occa::char4>() {
      return char4;
    }

    template <> dtype_t get<occa::ushort2>() {
      return ushort2;
    }

    template <> dtype_t get<occa::ushort4>() {
      return ushort4;
    }

    template <> dtype_t get<occa::short2>() {
      return short2;
    }

    template <> dtype_t get<occa::short4>() {
      return short4;
    }

    template <> dtype_t get<occa::uint2>() {
      return uint2;
    }

    template <> dtype_t get<occa::uint4>() {
      return uint4;
    }

    template <> dtype_t get<occa::int2>() {
      return int2;
    }

    template <> dtype_t get<occa::int4>() {
      return int4;
    }

    template <> dtype_t get<occa::ulong2>() {
      return ulong2;
    }

    template <> dtype_t get<occa::ulong4>() {
      return ulong4;
    }

    template <> dtype_t get<occa::long2>() {
      return long2;
    }

    template <> dtype_t get<occa::long4>() {
      return long4;
    }

    template <> dtype_t get<occa::float2>() {
      return float2;
    }

    template <> dtype_t get<occa::float4>() {
      return float4;
    }

    template <> dtype_t get<occa::double2>() {
      return double2;
    }

    template <> dtype_t get<occa::double4>() {
      return double4;
    }

    // OCCA Types
    template <> dtype_t get<occa::memory>() {
      return memory;
    }
  }
}
