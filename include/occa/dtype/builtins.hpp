#ifndef OCCA_DTYPE_BUILTINS_HEADER
#define OCCA_DTYPE_BUILTINS_HEADER

#include <occa/dtype/dtype.hpp>
#include <occa/types/typedefs.hpp>
#include <occa/types/typeinfo.hpp>
#include <occa/types/tuples.hpp>

namespace occa {
  class memory;

  typedef std::vector<dtype_t> dtypeVector;

  namespace dtype {
    extern const dtype_t none;

    extern const dtype_t void_;
    extern const dtype_t byte;

    extern const dtype_t bool_;
    extern const dtype_t char_;
    extern const dtype_t short_;
    extern const dtype_t int_;
    extern const dtype_t long_;
    extern const dtype_t ulong_;
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

    // OCCA Types
    extern const dtype_t memory;

    // Templated types
    template <class T>
    dtype_t get() {
      if (!typeMetadata<T>::isPointer) {
        return none;
      }
      return get<typename typeMetadata<T>::baseType>();
    }

    template <class T = void, class ...Types>
    dtypeVector getMany() {
      dtypeVector types = { get<T>() };

      dtypeVector tail = getMany<Types...>();
      types.insert(types.end(), tail.begin(), tail.end());

      return types;
    }

    template <>
    inline dtypeVector getMany() {
      return {};
    }

    // Primitive types
    template <> dtype_t get<void>();
    template <> dtype_t get<bool>();
    template <> dtype_t get<char>();
    template <> dtype_t get<signed char>();
    template <> dtype_t get<unsigned char>();
    template <> dtype_t get<short>();
    template <> dtype_t get<unsigned short>();
    template <> dtype_t get<int>();
    template <> dtype_t get<unsigned int>();
    template <> dtype_t get<long>();
    template <> dtype_t get<unsigned long>();
    template <> dtype_t get<long long>();
    template <> dtype_t get<unsigned long long>();
    template <> dtype_t get<float>();
    template <> dtype_t get<double>();
    template <> dtype_t get<occa::uchar2>();
    template <> dtype_t get<occa::uchar4>();
    template <> dtype_t get<occa::char2>();
    template <> dtype_t get<occa::char4>();
    template <> dtype_t get<occa::ushort2>();
    template <> dtype_t get<occa::ushort4>();
    template <> dtype_t get<occa::short2>();
    template <> dtype_t get<occa::short4>();
    template <> dtype_t get<occa::uint2>();
    template <> dtype_t get<occa::uint4>();
    template <> dtype_t get<occa::int2>();
    template <> dtype_t get<occa::int4>();
    template <> dtype_t get<occa::ulong2>();
    template <> dtype_t get<occa::ulong4>();
    template <> dtype_t get<occa::long2>();
    template <> dtype_t get<occa::long4>();
    template <> dtype_t get<occa::float2>();
    template <> dtype_t get<occa::float4>();
    template <> dtype_t get<occa::double2>();
    template <> dtype_t get<occa::double4>();

    // OCCA Types
    template <> dtype_t get<occa::memory>();
  }
}

#endif
