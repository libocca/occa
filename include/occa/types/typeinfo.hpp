#ifndef OCCA_TYPES_TYPEINFO_HEADER
#define OCCA_TYPES_TYPEINFO_HEADER

#include <occa/types/typedefs.hpp>
#include <occa/types/tuples.hpp>

namespace occa {
  template <class T>
  struct typeMetadata {
    typedef T baseType;
    static const bool isPointer = false;
  };

  template <class T>
  struct typeMetadata<T*> {
    typedef T baseType;
    static const bool isPointer = true;
  };

  template <class T>
  class primitiveinfo {
  public:
    static const std::string id;
    static const std::string name;
    static const bool isUnsigned;
  };

  template <class T>
  class typeinfo {
  public:
    static const std::string id;
    static const std::string name;
    static const bool isUnsigned;
  };

  typedef signed char    schar_t;
  typedef unsigned char  uchar_t;
  typedef unsigned short ushort_t;
  typedef unsigned int   uint_t;
  typedef unsigned long  ulong_t;

#ifndef _MSC_VER
  // NBN: these are defined in /occa/src/types/typeinfo.cpp
  template <> const std::string primitiveinfo<char>::id;
  template <> const std::string primitiveinfo<char>::name;
  template <> const bool        primitiveinfo<char>::isUnsigned;

  template <> const std::string primitiveinfo<short>::id;
  template <> const std::string primitiveinfo<short>::name;
  template <> const bool        primitiveinfo<short>::isUnsigned;

  template <> const std::string primitiveinfo<int>::id;
  template <> const std::string primitiveinfo<int>::name;
  template <> const bool        primitiveinfo<int>::isUnsigned;

  template <> const std::string primitiveinfo<long>::id;
  template <> const std::string primitiveinfo<long>::name;
  template <> const bool        primitiveinfo<long>::isUnsigned;

  template <> const std::string primitiveinfo<schar_t>::id;
  template <> const std::string primitiveinfo<schar_t>::name;
  template <> const bool        primitiveinfo<schar_t>::isUnsigned;

  template <> const std::string primitiveinfo<uchar_t>::id;
  template <> const std::string primitiveinfo<uchar_t>::name;
  template <> const bool        primitiveinfo<uchar_t>::isUnsigned;

  template <> const std::string primitiveinfo<ushort_t>::id;
  template <> const std::string primitiveinfo<ushort_t>::name;
  template <> const bool        primitiveinfo<ushort_t>::isUnsigned;

  template <> const std::string primitiveinfo<uint_t>::id;
  template <> const std::string primitiveinfo<uint_t>::name;
  template <> const bool        primitiveinfo<uint_t>::isUnsigned;

  template <> const std::string primitiveinfo<ulong_t>::id;
  template <> const std::string primitiveinfo<ulong_t>::name;
  template <> const bool        primitiveinfo<ulong_t>::isUnsigned;

  template <> const std::string primitiveinfo<float>::id;
  template <> const std::string primitiveinfo<float>::name;
  template <> const bool        primitiveinfo<float>::isUnsigned;

  template <> const std::string primitiveinfo<double>::id;
  template <> const std::string primitiveinfo<double>::name;
  template <> const bool        primitiveinfo<double>::isUnsigned;

  template <> const std::string typeinfo<uint8_t>::id;
  template <> const std::string typeinfo<uint8_t>::name;
  template <> const bool        typeinfo<uint8_t>::isUnsigned;

  template <> const std::string typeinfo<uint16_t>::id;
  template <> const std::string typeinfo<uint16_t>::name;
  template <> const bool        typeinfo<uint16_t>::isUnsigned;

  template <> const std::string typeinfo<uint32_t>::id;
  template <> const std::string typeinfo<uint32_t>::name;
  template <> const bool        typeinfo<uint32_t>::isUnsigned;

  template <> const std::string typeinfo<uint64_t>::id;
  template <> const std::string typeinfo<uint64_t>::name;
  template <> const bool        typeinfo<uint64_t>::isUnsigned;

  template <> const std::string typeinfo<int8_t>::id;
  template <> const std::string typeinfo<int8_t>::name;
  template <> const bool        typeinfo<int8_t>::isUnsigned;

  template <> const std::string typeinfo<int16_t>::id;
  template <> const std::string typeinfo<int16_t>::name;
  template <> const bool        typeinfo<int16_t>::isUnsigned;

  template <> const std::string typeinfo<int32_t>::id;
  template <> const std::string typeinfo<int32_t>::name;
  template <> const bool        typeinfo<int32_t>::isUnsigned;

  template <> const std::string typeinfo<int64_t>::id;
  template <> const std::string typeinfo<int64_t>::name;
  template <> const bool        typeinfo<int64_t>::isUnsigned;

  template <> const std::string typeinfo<float>::id;
  template <> const std::string typeinfo<float>::name;
  template <> const bool        typeinfo<float>::isUnsigned;

  template <> const std::string typeinfo<double>::id;
  template <> const std::string typeinfo<double>::name;
  template <> const bool        typeinfo<double>::isUnsigned;

  template <> const std::string typeinfo<uchar2>::id;
  template <> const std::string typeinfo<uchar2>::name;
  template <> const bool        typeinfo<uchar2>::isUnsigned;

  template <> const std::string typeinfo<uchar4>::id;
  template <> const std::string typeinfo<uchar4>::name;
  template <> const bool        typeinfo<uchar4>::isUnsigned;

  template <> const std::string typeinfo<char2>::id;
  template <> const std::string typeinfo<char2>::name;
  template <> const bool        typeinfo<char2>::isUnsigned;

  template <> const std::string typeinfo<char4>::id;
  template <> const std::string typeinfo<char4>::name;
  template <> const bool        typeinfo<char4>::isUnsigned;

  template <> const std::string typeinfo<ushort2>::id;
  template <> const std::string typeinfo<ushort2>::name;
  template <> const bool        typeinfo<ushort2>::isUnsigned;

  template <> const std::string typeinfo<ushort4>::id;
  template <> const std::string typeinfo<ushort4>::name;
  template <> const bool        typeinfo<ushort4>::isUnsigned;

  template <> const std::string typeinfo<short2>::id;
  template <> const std::string typeinfo<short2>::name;
  template <> const bool        typeinfo<short2>::isUnsigned;

  template <> const std::string typeinfo<short4>::id;
  template <> const std::string typeinfo<short4>::name;
  template <> const bool        typeinfo<short4>::isUnsigned;

  template <> const std::string typeinfo<uint2>::id;
  template <> const std::string typeinfo<uint2>::name;
  template <> const bool        typeinfo<uint2>::isUnsigned;

  template <> const std::string typeinfo<uint4>::id;
  template <> const std::string typeinfo<uint4>::name;
  template <> const bool        typeinfo<uint4>::isUnsigned;

  template <> const std::string typeinfo<int2>::id;
  template <> const std::string typeinfo<int2>::name;
  template <> const bool        typeinfo<int2>::isUnsigned;

  template <> const std::string typeinfo<int4>::id;
  template <> const std::string typeinfo<int4>::name;
  template <> const bool        typeinfo<int4>::isUnsigned;

  template <> const std::string typeinfo<ulong2>::id;
  template <> const std::string typeinfo<ulong2>::name;
  template <> const bool        typeinfo<ulong2>::isUnsigned;

  template <> const std::string typeinfo<ulong4>::id;
  template <> const std::string typeinfo<ulong4>::name;
  template <> const bool        typeinfo<ulong4>::isUnsigned;

  template <> const std::string typeinfo<long2>::id;
  template <> const std::string typeinfo<long2>::name;
  template <> const bool        typeinfo<long2>::isUnsigned;

  template <> const std::string typeinfo<long4>::id;
  template <> const std::string typeinfo<long4>::name;
  template <> const bool        typeinfo<long4>::isUnsigned;

  template <> const std::string typeinfo<float2>::id;
  template <> const std::string typeinfo<float2>::name;
  template <> const bool        typeinfo<float2>::isUnsigned;

  template <> const std::string typeinfo<float4>::id;
  template <> const std::string typeinfo<float4>::name;
  template <> const bool        typeinfo<float4>::isUnsigned;

  template <> const std::string typeinfo<double2>::id;
  template <> const std::string typeinfo<double2>::name;
  template <> const bool        typeinfo<double2>::isUnsigned;

  template <> const std::string typeinfo<double4>::id;
  template <> const std::string typeinfo<double4>::name;
  template <> const bool        typeinfo<double4>::isUnsigned;
#else
  // NBN: MSVC error C2737: "const object must be initialized"
  template <> const std::string primitiveinfo<char>::id         = "c";
  template <> const std::string primitiveinfo<char>::name       = "char";
  template <> const bool        primitiveinfo<char>::isUnsigned = false;

  template <> const std::string primitiveinfo<short>::id         = "s";
  template <> const std::string primitiveinfo<short>::name       = "short";
  template <> const bool        primitiveinfo<short>::isUnsigned = false;

  template <> const std::string primitiveinfo<int>::id         = "i";
  template <> const std::string primitiveinfo<int>::name       = "int";
  template <> const bool        primitiveinfo<int>::isUnsigned = false;

  template <> const std::string primitiveinfo<long>::id         = "l";
  template <> const std::string primitiveinfo<long>::name       = "long";
  template <> const bool        primitiveinfo<long>::isUnsigned = false;

  template <> const std::string primitiveinfo<schar_t>::id         = "sc";
  template <> const std::string primitiveinfo<schar_t>::name       = "schar_t";
  template <> const bool        primitiveinfo<schar_t>::isUnsigned = false;

  template <> const std::string primitiveinfo<uchar_t>::id         = "uc";
  template <> const std::string primitiveinfo<uchar_t>::name       = "uchar_t";
  template <> const bool        primitiveinfo<uchar_t>::isUnsigned = true;

  template <> const std::string primitiveinfo<ushort_t>::id         = "us";
  template <> const std::string primitiveinfo<ushort_t>::name       = "ushort_t";
  template <> const bool        primitiveinfo<ushort_t>::isUnsigned = true;

  template <> const std::string primitiveinfo<uint_t>::id         = "ui";
  template <> const std::string primitiveinfo<uint_t>::name       = "uint_t";
  template <> const bool        primitiveinfo<uint_t>::isUnsigned = true;

  template <> const std::string primitiveinfo<ulong_t>::id         = "ul";
  template <> const std::string primitiveinfo<ulong_t>::name       = "ulong_t";
  template <> const bool        primitiveinfo<ulong_t>::isUnsigned = true;

  template <> const std::string primitiveinfo<float>::id         = "f";
  template <> const std::string primitiveinfo<float>::name       = "float";
  template <> const bool        primitiveinfo<float>::isUnsigned = false;

  template <> const std::string primitiveinfo<double>::id         = "d";
  template <> const std::string primitiveinfo<double>::name       = "double";
  template <> const bool        primitiveinfo<double>::isUnsigned = false;

  template <> const std::string typeinfo<uint8_t>::id         = "u8";
  template <> const std::string typeinfo<uint8_t>::name       = "uint8";
  template <> const bool        typeinfo<uint8_t>::isUnsigned = true;

  template <> const std::string typeinfo<uint16_t>::id         = "u16";
  template <> const std::string typeinfo<uint16_t>::name       = "uint16";
  template <> const bool        typeinfo<uint16_t>::isUnsigned = true;

  template <> const std::string typeinfo<uint32_t>::id         = "u32";
  template <> const std::string typeinfo<uint32_t>::name       = "uint32";
  template <> const bool        typeinfo<uint32_t>::isUnsigned = true;

  template <> const std::string typeinfo<uint64_t>::id         = "u64";
  template <> const std::string typeinfo<uint64_t>::name       = "uint64";
  template <> const bool        typeinfo<uint64_t>::isUnsigned = true;

  template <> const std::string typeinfo<int8_t>::id         = "i8";
  template <> const std::string typeinfo<int8_t>::name       = "int8";
  template <> const bool        typeinfo<int8_t>::isUnsigned = false;

  template <> const std::string typeinfo<int16_t>::id         = "i16";
  template <> const std::string typeinfo<int16_t>::name       = "int16";
  template <> const bool        typeinfo<int16_t>::isUnsigned = false;

  template <> const std::string typeinfo<int32_t>::id         = "i32";
  template <> const std::string typeinfo<int32_t>::name       = "int32";
  template <> const bool        typeinfo<int32_t>::isUnsigned = false;

  template <> const std::string typeinfo<int64_t>::id         = "i64";
  template <> const std::string typeinfo<int64_t>::name       = "int64";
  template <> const bool        typeinfo<int64_t>::isUnsigned = false;

  template <> const std::string typeinfo<float>::id         = "f32";
  template <> const std::string typeinfo<float>::name       = "float";
  template <> const bool        typeinfo<float>::isUnsigned = false;

  template <> const std::string typeinfo<double>::id         = "f64";
  template <> const std::string typeinfo<double>::name       = "double";
  template <> const bool        typeinfo<double>::isUnsigned = false;

  template <> const std::string typeinfo<uchar2>::id         = "vuc2";
  template <> const std::string typeinfo<uchar2>::name       = "uchar2";
  template <> const bool        typeinfo<uchar2>::isUnsigned = true;

  template <> const std::string typeinfo<uchar4>::id         = "vuc4";
  template <> const std::string typeinfo<uchar4>::name       = "uchar4";
  template <> const bool        typeinfo<uchar4>::isUnsigned = true;

  template <> const std::string typeinfo<char2>::id         = "vc2";
  template <> const std::string typeinfo<char2>::name       = "char2";
  template <> const bool        typeinfo<char2>::isUnsigned = false;

  template <> const std::string typeinfo<char4>::id         = "vc4";
  template <> const std::string typeinfo<char4>::name       = "char4";
  template <> const bool        typeinfo<char4>::isUnsigned = false;

  template <> const std::string typeinfo<ushort2>::id         = "vus2";
  template <> const std::string typeinfo<ushort2>::name       = "ushort2";
  template <> const bool        typeinfo<ushort2>::isUnsigned = true;

  template <> const std::string typeinfo<ushort4>::id         = "vus4";
  template <> const std::string typeinfo<ushort4>::name       = "ushort4";
  template <> const bool        typeinfo<ushort4>::isUnsigned = true;

  template <> const std::string typeinfo<short2>::id         = "vs2";
  template <> const std::string typeinfo<short2>::name       = "short2";
  template <> const bool        typeinfo<short2>::isUnsigned = false;

  template <> const std::string typeinfo<short4>::id         = "vs4";
  template <> const std::string typeinfo<short4>::name       = "short4";
  template <> const bool        typeinfo<short4>::isUnsigned = false;

  template <> const std::string typeinfo<uint2>::id         = "vui2";
  template <> const std::string typeinfo<uint2>::name       = "uint2";
  template <> const bool        typeinfo<uint2>::isUnsigned = true;

  template <> const std::string typeinfo<uint4>::id         = "vui4";
  template <> const std::string typeinfo<uint4>::name       = "uint4";
  template <> const bool        typeinfo<uint4>::isUnsigned = true;

  template <> const std::string typeinfo<int2>::id         = "vi2";
  template <> const std::string typeinfo<int2>::name       = "int2";
  template <> const bool        typeinfo<int2>::isUnsigned = false;

  template <> const std::string typeinfo<int4>::id         = "vi4";
  template <> const std::string typeinfo<int4>::name       = "int4";
  template <> const bool        typeinfo<int4>::isUnsigned = false;

  template <> const std::string typeinfo<ulong2>::id         = "vul2";
  template <> const std::string typeinfo<ulong2>::name       = "ulong2";
  template <> const bool        typeinfo<ulong2>::isUnsigned = true;

  template <> const std::string typeinfo<ulong4>::id         = "vul4";
  template <> const std::string typeinfo<ulong4>::name       = "ulong4";
  template <> const bool        typeinfo<ulong4>::isUnsigned = true;

  template <> const std::string typeinfo<long2>::id         = "vl2";
  template <> const std::string typeinfo<long2>::name       = "long2";
  template <> const bool        typeinfo<long2>::isUnsigned = false;

  template <> const std::string typeinfo<long4>::id         = "vl4";
  template <> const std::string typeinfo<long4>::name       = "long4";
  template <> const bool        typeinfo<long4>::isUnsigned = false;

  template <> const std::string typeinfo<float2>::id         = "vf2";
  template <> const std::string typeinfo<float2>::name       = "float2";
  template <> const bool        typeinfo<float2>::isUnsigned = false;

  template <> const std::string typeinfo<float4>::id         = "vf4";
  template <> const std::string typeinfo<float4>::name       = "float4";
  template <> const bool        typeinfo<float4>::isUnsigned = false;

  template <> const std::string typeinfo<double2>::id         = "vd2";
  template <> const std::string typeinfo<double2>::name       = "double2";
  template <> const bool        typeinfo<double2>::isUnsigned = false;

  template <> const std::string typeinfo<double4>::id         = "vd4";
  template <> const std::string typeinfo<double4>::name       = "double4";
  template <> const bool        typeinfo<double4>::isUnsigned = false;
#endif  // NBN: MSVC

}

#endif
