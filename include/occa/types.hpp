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

#ifndef OCCA_TYPES_HEADER
#define OCCA_TYPES_HEADER

#include <iostream>
#include <vector>
#include <map>
#include <stdint.h>

#include "occa/defines.hpp"
#include "occa/vector.hpp"

namespace occa {
  typedef int64_t dim_t;
  typedef uint64_t udim_t;

  typedef std::vector<int>                   intVector_t;
  typedef std::vector<intVector_t>           intVecVector_t;

  typedef std::vector<std::string>           strVector_t;
  typedef strVector_t::iterator              strVectorIterator;
  typedef strVector_t::const_iterator        cStrVectorIterator;

  typedef std::map<std::string, std::string> strToStrMap_t;
  typedef strToStrMap_t::iterator            strToStrMapIterator;
  typedef strToStrMap_t::const_iterator      cStrToStrMapIterator;

  typedef std::map<std::string,strVector_t>  strToStrsMap_t;
  typedef strToStrsMap_t::iterator           strToStrsMapIterator;
  typedef strToStrsMap_t::const_iterator     cStrToStrsMapIterator;

  typedef std::map<std::string, bool>        strToBoolMap_t;
  typedef strToBoolMap_t::iterator           strToBoolMapIterator;
  typedef strToBoolMap_t::const_iterator     cStrToBoolMapIterator;

  //---[ Dim ]--------------------------
  class dim {
  public:
    int dims;
    udim_t x, y, z;

    dim();
    dim(udim_t x_);
    dim(udim_t x_, udim_t y_);
    dim(udim_t x_, udim_t y_, udim_t z_);
    dim(int dims_, udim_t x_, udim_t y_, udim_t z_);

    dim(const dim &d);

    dim& operator = (const dim &d);

    dim operator + (const dim &d) const;
    dim operator - (const dim &d) const;
    dim operator * (const dim &d) const;
    dim operator / (const dim &d) const;

    bool hasNegativeEntries();

    udim_t& operator [] (int i);
    udim_t  operator [] (int i) const;
  };
  //====================================

  //---[ Type To String ]---------------
  template <class TM>
  class primitiveinfo {
  public:
    static const std::string id;
    static const std::string name;
    static const bool isUnsigned;
  };

  template <class TM>
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
  //====================================
}

#endif
