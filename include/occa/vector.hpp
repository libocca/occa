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

#ifndef OCCA_VECTOR_DEFINE_HEADER
#define OCCA_VECTOR_DEFINE_HEADER

#include "occa/defines.hpp"

#if OCCA_MMX
#  include <mmintrin.h>
#endif

#if OCCA_SSE
#  include <xmmintrin.h>
#endif

#if OCCA_SSE2
#  include <emmintrin.h>
#endif

#if OCCA_SSE3
#  include <pmmintrin.h>
#endif

#if OCCA_SSSE3
#  include <tmmintrin.h>
#endif

#if OCCA_SSE4_1
#  include <smmintrin.h>
#endif

#if OCCA_SSE4_2
#  include <nmmintrin.h>
#endif

#if OCCA_AVX
#  include <immintrin.h>
#endif

namespace occa {
  //---[ type2 ]------------------------
  template <class TM>
  class type2 {
  public:
    union { TM s0, x; };
    union { TM s1, y; };

    inline type2() :
      x(0),
      y(0) {}

    inline type2(const TM &x_) :
      x(x_),
      y(0) {}

    inline type2(const TM &x_,
                 const TM &y_) :
      x(x_),
      y(y_) {}
  };

  template <class TM>
  inline type2<TM> operator - (const type2<TM> &a) {
    return type2<TM>(-a.x, -a.y);
  }

  template <class TM>
  inline type2<TM> operator + (const type2<TM> &a, const type2<TM> &b) {
    return type2<TM>(a.x + b.x,
                     a.y + b.y);
  }

  template <class TM>
  inline type2<TM> operator + (const TM a, const type2<TM> &b) {
    return type2<TM>(a + b.x,
                     a + b.y);
  }

  template <class TM>
  inline type2<TM> operator + (const type2<TM> &a, const TM b) {
    return type2<TM>(a.x + b,
                     a.y + b);
  }

  template <class TM>
  inline type2<TM>& operator += ( type2<TM> &a, const type2<TM> &b) {
    a.x += b.x;
    a.y += b.y;
    return a;
  }

  template <class TM>
  inline type2<TM>& operator += ( type2<TM> &a, const TM b) {
    a.x += b;
    a.y += b;
    return a;
  }

  template <class TM>
  inline type2<TM> operator - (const type2<TM> &a, const type2<TM> &b) {
    return type2<TM>(a.x - b.x,
                     a.y - b.y);
  }

  template <class TM>
  inline type2<TM> operator - (const TM a, const type2<TM> &b) {
    return type2<TM>(a - b.x,
                     a - b.y);
  }

  template <class TM>
  inline type2<TM> operator - (const type2<TM> &a, const TM b) {
    return type2<TM>(a.x - b,
                     a.y - b);
  }

  template <class TM>
  inline type2<TM>& operator -= ( type2<TM> &a, const type2<TM> &b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
  }

  template <class TM>
  inline type2<TM>& operator -= ( type2<TM> &a, const TM b) {
    a.x -= b;
    a.y -= b;
    return a;
  }

  template <class TM>
  inline type2<TM> operator * (const type2<TM> &a, const type2<TM> &b) {
    return type2<TM>(a.x * b.x,
                     a.y * b.y);
  }

  template <class TM>
  inline type2<TM> operator * (const TM a, const type2<TM> &b) {
    return type2<TM>(a * b.x,
                     a * b.y);
  }

  template <class TM>
  inline type2<TM> operator * (const type2<TM> &a, const TM b) {
    return type2<TM>(a.x * b,
                     a.y * b);
  }

  template <class TM>
  inline type2<TM>& operator *= ( type2<TM> &a, const type2<TM> &b) {
    a.x *= b.x;
    a.y *= b.y;
    return a;
  }

  template <class TM>
  inline type2<TM>& operator *= ( type2<TM> &a, const TM b) {
    a.x *= b;
    a.y *= b;
    return a;
  }

  template <class TM>
  inline type2<TM> operator / (const type2<TM> &a, const type2<TM> &b) {
    return type2<TM>(a.x / b.x,
                     a.y / b.y);
  }

  template <class TM>
  inline type2<TM> operator / (const TM a, const type2<TM> &b) {
    return type2<TM>(a / b.x,
                     a / b.y);
  }

  template <class TM>
  inline type2<TM> operator / (const type2<TM> &a, const TM b) {
    return type2<TM>(a.x / b,
                     a.y / b);
  }

  template <class TM>
  inline type2<TM>& operator /= ( type2<TM> &a, const type2<TM> &b) {
    a.x /= b.x;
    a.y /= b.y;
    return a;
  }

  template <class TM>
  inline type2<TM>& operator /= ( type2<TM> &a, const TM b) {
    a.x /= b;
    a.y /= b;
    return a;
  }

  template <class TM>
  inline std::ostream& operator << (std::ostream &out, const type2<TM>& a) {
    out << "["
        << a.x << ", "
        << a.y
        << "]\n";
    return out;
  }
  //====================================

  //---[ type4 ]------------------------
  template <class TM>
  class type4 {
  public:
    union { TM s0, x; };
    union { TM s1, y; };
    union { TM s2, z; };
    union { TM s3, w; };

    inline type4() :
      x(0),
      y(0),
      z(0),
      w(0) {}

    inline type4(const TM &x_) :
      x(x_),
      y(0),
      z(0),
      w(0) {}

    inline type4(const TM &x_,
                 const TM &y_) :
      x(x_),
      y(y_),
      z(0),
      w(0) {}

    inline type4(const TM &x_,
                 const TM &y_,
                 const TM &z_) :
      x(x_),
      y(y_),
      z(z_),
      w(0) {}

    inline type4(const TM &x_,
                 const TM &y_,
                 const TM &z_,
                 const TM &w_) :
      x(x_),
      y(y_),
      z(z_),
      w(w_) {}
  };

  template <class TM>
  inline type4<TM> operator - (const type4<TM> &a) {
    return type4<TM>(-a.x, -a.y, -a.z, -a.w);
  }

  template <class TM>
  inline type4<TM> operator + (const type4<TM> &a, const type4<TM> &b) {
    return type4<TM>(a.x + b.x,
                     a.y + b.y,
                     a.z + b.z,
                     a.w + b.w);
  }

  template <class TM, class TM2>
  inline type4<TM> operator + (const TM2 a, const type4<TM> &b) {
    return type4<TM>(a + b.x,
                     a + b.y,
                     a + b.z,
                     a + b.w);
  }

  template <class TM, class TM2>
  inline type4<TM> operator + (const type4<TM> &a, const TM2 b) {
    return type4<TM>(a.x + b,
                     a.y + b,
                     a.z + b,
                     a.w + b);
  }

  template <class TM>
  inline type4<TM>& operator += ( type4<TM> &a, const type4<TM> &b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
  }

  template <class TM, class TM2>
  inline type4<TM>& operator += ( type4<TM> &a, const TM2 b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
    return a;
  }

  template <class TM>
  inline type4<TM> operator - (const type4<TM> &a, const type4<TM> &b) {
    return type4<TM>(a.x - b.x,
                     a.y - b.y,
                     a.z - b.z,
                     a.w - b.w);
  }

  template <class TM, class TM2>
  inline type4<TM> operator - (const TM2 a, const type4<TM> &b) {
    return type4<TM>(a - b.x,
                     a - b.y,
                     a - b.z,
                     a - b.w);
  }

  template <class TM, class TM2>
  inline type4<TM> operator - (const type4<TM> &a, const TM2 b) {
    return type4<TM>(a.x - b,
                     a.y - b,
                     a.z - b,
                     a.w - b);
  }

  template <class TM>
  inline type4<TM>& operator -= ( type4<TM> &a, const type4<TM> &b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
  }

  template <class TM, class TM2>
  inline type4<TM>& operator -= ( type4<TM> &a, const TM2 b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
    return a;
  }

  template <class TM>
  inline type4<TM> operator * (const type4<TM> &a, const type4<TM> &b) {
    return type4<TM>(a.x * b.x,
                     a.y * b.y,
                     a.z * b.z,
                     a.w * b.w);
  }

  template <class TM, class TM2>
  inline type4<TM> operator * (const TM2 a, const type4<TM> &b) {
    return type4<TM>(a * b.x,
                     a * b.y,
                     a * b.z,
                     a * b.w);
  }

  template <class TM, class TM2>
  inline type4<TM> operator * (const type4<TM> &a, const TM2 b) {
    return type4<TM>(a.x * b,
                     a.y * b,
                     a.z * b,
                     a.w * b);
  }

  template <class TM>
  inline type4<TM>& operator *= ( type4<TM> &a, const type4<TM> &b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    return a;
  }

  template <class TM, class TM2>
  inline type4<TM>& operator *= ( type4<TM> &a, const TM2 b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    return a;
  }

  template <class TM>
  inline type4<TM> operator / (const type4<TM> &a, const type4<TM> &b) {
    return type4<TM>(a.x / b.x,
                     a.y / b.y,
                     a.z / b.z,
                     a.w / b.w);
  }

  template <class TM, class TM2>
  inline type4<TM> operator / (const TM2 a, const type4<TM> &b) {
    return type4<TM>(a / b.x,
                     a / b.y,
                     a / b.z,
                     a / b.w);
  }

  template <class TM, class TM2>
  inline type4<TM> operator / (const type4<TM> &a, const TM2 b) {
    return type4<TM>(a.x / b,
                     a.y / b,
                     a.z / b,
                     a.w / b);
  }

  template <class TM>
  inline type4<TM>& operator /= ( type4<TM> &a, const type4<TM> &b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
    return a;
  }

  template <class TM, class TM2>
  inline type4<TM>& operator /= ( type4<TM> &a, const TM2 b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
    return a;
  }

  template <class TM>
  inline std::ostream& operator << (std::ostream &out, const type4<TM>& a) {
    out << "["
        << a.x << ", "
        << a.y << ", "
        << a.z << ", "
        << a.w
        << "]\n";
    return out;
  }
  //====================================

  //---[ Typedefs ]---------------------
  typedef type2<unsigned char> uchar2;
  typedef type4<unsigned char> uchar3;
  typedef type4<unsigned char> uchar4;

  typedef type2<char> char2;
  typedef type4<char> char3;
  typedef type4<char> char4;

  typedef type2<unsigned short> ushort2;
  typedef type4<unsigned short> ushort3;
  typedef type4<unsigned short> ushort4;

  typedef type2<short> short2;
  typedef type4<short> short3;
  typedef type4<short> short4;

  typedef type2<unsigned int> uint2;
  typedef type4<unsigned int> uint3;
  typedef type4<unsigned int> uint4;

  typedef type2<int> int2;
  typedef type4<int> int3;
  typedef type4<int> int4;

  typedef type2<unsigned long> ulong2;
  typedef type4<unsigned long> ulong3;
  typedef type4<unsigned long> ulong4;

  typedef type2<long> long2;
  typedef type4<long> long3;
  typedef type4<long> long4;

  typedef type2<float> float2;
  typedef type4<float> float3;
  typedef type4<float> float4;

  typedef type2<double> double2;
  typedef type4<double> double3;
  typedef type4<double> double4;
  //====================================

  //---[ Functions ]--------------------
  template <class TM>
  inline double length(const type2<TM> &v) {
    return sqrt((v.x * v.x) +
                (v.y * v.y));
  }

  template <class TM>
  inline double length(const type4<TM> &v) {
    return sqrt((v.x * v.x) +
                (v.y * v.y) +
                (v.z * v.z) +
                (v.w * v.w));
  }

  template <class TM>
  inline type2<TM> normalize(const type2<TM> &v) {
    const double invNorm = (1.0 / length(v));
    return type2<TM>(invNorm * v.x,
                     invNorm * v.y);
  }

  template <class TM>
  inline type4<TM> normalize(const type4<TM> &v) {
    const double invNorm = (1.0 / length(v));
    return type4<TM>(invNorm * v.x,
                     invNorm * v.y,
                     invNorm * v.z,
                     invNorm * v.w);
  }

  template <class TM>
  inline double dot(const type2<TM> &a, const type2<TM> &b) {
    return ((a.x * b.x) +
            (a.y * b.y));
  }

  template <class TM>
  inline double dot(const type4<TM> &a, const type4<TM> &b) {
    return ((a.x * b.x) +
            (a.y * b.y) +
            (a.z * b.z) +
            (a.w * b.w));
  }

  template <class TM>
  inline TM clamp(const TM &value, const TM min, const TM max) {
    return (value < min) ? min : ((max < value) ? max : value);
  }

  template <class TM, class TM2>
  inline type2<TM> clamp(const type2<TM> &v, const TM2 min, const TM2 max) {
    return type2<TM>(clamp(v.x, min, max),
                     clamp(v.y, min, max));
  }

  template <class TM, class TM2>
  inline type4<TM> clamp(const type4<TM> &v, const TM2 min, const TM2 max) {
    return type4<TM>(clamp(v.x, min, max),
                     clamp(v.y, min, max),
                     clamp(v.z, min, max),
                     clamp(v.w, min, max));
  }

  inline float4 cross(const float4 &a, const float4 &b) {
    return float4((a.z * b.y) - (b.z * a.y),
                  (a.x * b.z) - (b.x * a.z),
                  (a.y * b.x) - (b.y * a.x));
  }

  inline double4 cross(const double4 &a, const double4 &b) {
    return double3((a.z * b.y) - (b.z * a.y),
                   (a.x * b.z) - (b.x * a.z),
                   (a.y * b.x) - (b.y * a.x));
  }
  //====================================
}

#endif
