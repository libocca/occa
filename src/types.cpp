/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#include "occa/types.hpp"

namespace occa {
  //---[ Dim ]--------------------------
  dim::dim() :
    dims(0),
    x(1),
    y(1),
    z(1) {}

  dim::dim(udim_t x_) :
    dims(1),
    x(x_),
    y(1),
    z(1) {}

  dim::dim(udim_t x_, udim_t y_) :
    dims(2),
    x(x_),
    y(y_),
    z(1) {}

  dim::dim(udim_t x_, udim_t y_, udim_t z_) :
    dims(3),
    x(x_),
    y(y_),
    z(z_) {}

  dim::dim(int dims_, udim_t x_, udim_t y_, udim_t z_) :
    dims(dims_),
    x(x_),
    y(y_),
    z(z_) {}

  dim::dim(const dim &d) :
    dims(d.dims),
    x(d.x),
    y(d.y),
    z(d.z) {}

  dim& dim::operator = (const dim &d) {
    dims = d.dims;
    x = d.x;
    y = d.y;
    z = d.z;
    return *this;
  }

  dim dim::operator + (const dim &d) const {
    return dim(dims > d.dims ? dims : d.dims,
               x + d.x,
               y + d.y,
               z + d.z);
  }

  dim dim::operator - (const dim &d) const {
    return dim(dims > d.dims ? dims : d.dims,
               x - d.x,
               y - d.y,
               z - d.z);
  }

  dim dim::operator * (const dim &d) const {
    return dim(dims > d.dims ? dims : d.dims,
               x * d.x,
               y * d.y,
               z * d.z);
  }

  dim dim::operator / (const dim &d) const {
    return dim(dims > d.dims ? dims : d.dims,
               x / d.x,
               y / d.y,
               z / d.z);
  }

  bool dim::hasNegativeEntries() {
    return ((x & (1 << (sizeof(udim_t) - 1))) ||
            (y & (1 << (sizeof(udim_t) - 1))) ||
            (z & (1 << (sizeof(udim_t) - 1))));
  }

  udim_t& dim::operator [] (int i) {
    switch(i) {
    case 0 : return x;
    case 1 : return y;
    default: return z;
    }
  }

  udim_t dim::operator [] (int i) const {
    switch(i) {
    case 0 : return x;
    case 1 : return y;
    default: return z;
    }
  }
  //====================================

  //---[ Type To String ]---------------
  template <>
  const std::string primitiveinfo<char>::id = "c";
  template <>
  const std::string primitiveinfo<char>::name = "char";
  template <>
  const bool primitiveinfo<char>::isUnsigned = false;

  template <>
  const std::string primitiveinfo<short>::id = "s";
  template <>
  const std::string primitiveinfo<short>::name = "short";
  template <>
  const bool primitiveinfo<short>::isUnsigned = false;

  template <>
  const std::string primitiveinfo<int>::id = "i";
  template <>
  const std::string primitiveinfo<int>::name = "int";
  template <>
  const bool primitiveinfo<int>::isUnsigned = false;

  template <>
  const std::string primitiveinfo<long>::id = "l";
  template <>
  const std::string primitiveinfo<long>::name = "long";
  template <>
  const bool primitiveinfo<long>::isUnsigned = false;

  template <>
  const std::string primitiveinfo<unsigned char>::id = "uc";
  template <>
  const std::string primitiveinfo<unsigned char>::name = "unsigned char";
  template <>
  const bool primitiveinfo<unsigned char>::isUnsigned = true;

  template <>
  const std::string primitiveinfo<unsigned short>::id = "us";
  template <>
  const std::string primitiveinfo<unsigned short>::name = "unsigned short";
  template <>
  const bool primitiveinfo<unsigned short>::isUnsigned = true;

  template <>
  const std::string primitiveinfo<unsigned int>::id = "ui";
  template <>
  const std::string primitiveinfo<unsigned int>::name = "unsigned int";
  template <>
  const bool primitiveinfo<unsigned int>::isUnsigned = true;

  template <>
  const std::string primitiveinfo<unsigned long>::id = "ul";
  template <>
  const std::string primitiveinfo<unsigned long>::name = "unsigned long";
  template <>
  const bool primitiveinfo<unsigned long>::isUnsigned = true;

  template <>
  const std::string primitiveinfo<signed char>::id = "sc";
  template <>
  const std::string primitiveinfo<signed char>::name = "signed char";
  template <>
  const bool primitiveinfo<signed char>::isUnsigned = false;

  template <>
  const std::string primitiveinfo<float>::id = "f";
  template <>
  const std::string primitiveinfo<float>::name = "float";
  template <>
  const bool primitiveinfo<float>::isUnsigned = false;

  template <>
  const std::string primitiveinfo<double>::id = "d";
  template <>
  const std::string primitiveinfo<double>::name = "double";
  template <>
  const bool primitiveinfo<double>::isUnsigned = false;

  template <>
  const std::string typeinfo<uint8_t>::id = "u8";
  template <>
  const std::string typeinfo<uint8_t>::name = "uint8";
  template <>
  const bool typeinfo<uint8_t>::isUnsigned = true;

  template <>
  const std::string typeinfo<uint16_t>::id = "u16";
  template <>
  const std::string typeinfo<uint16_t>::name = "uint16";
  template <>
  const bool typeinfo<uint16_t>::isUnsigned = true;

  template <>
  const std::string typeinfo<uint32_t>::id = "u32";
  template <>
  const std::string typeinfo<uint32_t>::name = "uint32";
  template <>
  const bool typeinfo<uint32_t>::isUnsigned = true;

  template <>
  const std::string typeinfo<uint64_t>::id = "u64";
  template <>
  const std::string typeinfo<uint64_t>::name = "uint64";
  template <>
  const bool typeinfo<uint64_t>::isUnsigned = true;

  template <>
  const std::string typeinfo<int8_t>::id = "i8";
  template <>
  const std::string typeinfo<int8_t>::name = "int8";
  template <>
  const bool typeinfo<int8_t>::isUnsigned = false;

  template <>
  const std::string typeinfo<int16_t>::id = "i16";
  template <>
  const std::string typeinfo<int16_t>::name = "int16";
  template <>
  const bool typeinfo<int16_t>::isUnsigned = false;

  template <>
  const std::string typeinfo<int32_t>::id = "i32";
  template <>
  const std::string typeinfo<int32_t>::name = "int32";
  template <>
  const bool typeinfo<int32_t>::isUnsigned = false;

  template <>
  const std::string typeinfo<int64_t>::id = "i64";
  template <>
  const std::string typeinfo<int64_t>::name = "int64";
  template <>
  const bool typeinfo<int64_t>::isUnsigned = false;

  template <>
  const std::string typeinfo<float>::id = "f32";
  template <>
  const std::string typeinfo<float>::name = "float";
  template <>
  const bool typeinfo<float>::isUnsigned = false;

  template <>
  const std::string typeinfo<double>::id = "f64";
  template <>
  const std::string typeinfo<double>::name = "double";
  template <>
  const bool typeinfo<double>::isUnsigned = false;

  template <>
  const std::string typeinfo<uchar2>::id = "vuc2";
  template <>
  const std::string typeinfo<uchar2>::name = "uchar2";
  template <>
  const bool typeinfo<uchar2>::isUnsigned = true;

  template <>
  const std::string typeinfo<uchar4>::id = "vuc4";
  template <>
  const std::string typeinfo<uchar4>::name = "uchar4";
  template <>
  const bool typeinfo<uchar4>::isUnsigned = true;

  template <>
  const std::string typeinfo<char2>::id = "vc2";
  template <>
  const std::string typeinfo<char2>::name = "char2";
  template <>
  const bool typeinfo<char2>::isUnsigned = false;

  template <>
  const std::string typeinfo<char4>::id = "vc4";
  template <>
  const std::string typeinfo<char4>::name = "char4";
  template <>
  const bool typeinfo<char4>::isUnsigned = false;

  template <>
  const std::string typeinfo<ushort2>::id = "vus2";
  template <>
  const std::string typeinfo<ushort2>::name = "ushort2";
  template <>
  const bool typeinfo<ushort2>::isUnsigned = true;

  template <>
  const std::string typeinfo<ushort4>::id = "vus4";
  template <>
  const std::string typeinfo<ushort4>::name = "ushort4";
  template <>
  const bool typeinfo<ushort4>::isUnsigned = true;

  template <>
  const std::string typeinfo<short2>::id = "vs2";
  template <>
  const std::string typeinfo<short2>::name = "short2";
  template <>
  const bool typeinfo<short2>::isUnsigned = false;

  template <>
  const std::string typeinfo<short4>::id = "vs4";
  template <>
  const std::string typeinfo<short4>::name = "short4";
  template <>
  const bool typeinfo<short4>::isUnsigned = false;

  template <>
  const std::string typeinfo<uint2>::id = "vui2";
  template <>
  const std::string typeinfo<uint2>::name = "uint2";
  template <>
  const bool typeinfo<uint2>::isUnsigned = true;

  template <>
  const std::string typeinfo<uint4>::id = "vui4";
  template <>
  const std::string typeinfo<uint4>::name = "uint4";
  template <>
  const bool typeinfo<uint4>::isUnsigned = true;

  template <>
  const std::string typeinfo<int2>::id = "vi2";
  template <>
  const std::string typeinfo<int2>::name = "int2";
  template <>
  const bool typeinfo<int2>::isUnsigned = false;

  template <>
  const std::string typeinfo<int4>::id = "vi4";
  template <>
  const std::string typeinfo<int4>::name = "int4";
  template <>
  const bool typeinfo<int4>::isUnsigned = false;

  template <>
  const std::string typeinfo<ulong2>::id = "vul2";
  template <>
  const std::string typeinfo<ulong2>::name = "ulong2";
  template <>
  const bool typeinfo<ulong2>::isUnsigned = true;

  template <>
  const std::string typeinfo<ulong4>::id = "vul4";
  template <>
  const std::string typeinfo<ulong4>::name = "ulong4";
  template <>
  const bool typeinfo<ulong4>::isUnsigned = true;

  template <>
  const std::string typeinfo<long2>::id = "vl2";
  template <>
  const std::string typeinfo<long2>::name = "long2";
  template <>
  const bool typeinfo<long2>::isUnsigned = false;

  template <>
  const std::string typeinfo<long4>::id = "vl4";
  template <>
  const std::string typeinfo<long4>::name = "long4";
  template <>
  const bool typeinfo<long4>::isUnsigned = false;

  template <>
  const std::string typeinfo<float2>::id = "vf2";
  template <>
  const std::string typeinfo<float2>::name = "float2";
  template <>
  const bool typeinfo<float2>::isUnsigned = false;

  template <>
  const std::string typeinfo<float4>::id = "vf4";
  template <>
  const std::string typeinfo<float4>::name = "float4";
  template <>
  const bool typeinfo<float4>::isUnsigned = false;

  template <>
  const std::string typeinfo<double2>::id = "vd2";
  template <>
  const std::string typeinfo<double2>::name = "double2";
  template <>
  const bool typeinfo<double2>::isUnsigned = false;

  template <>
  const std::string typeinfo<double4>::id = "vd4";
  template <>
  const std::string typeinfo<double4>::name = "double4";
  template <>
  const bool typeinfo<double4>::isUnsigned = false;
  //====================================
}
