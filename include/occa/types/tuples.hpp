#ifndef OCCA_TYPE_TUPLES_HEADER
#define OCCA_TYPE_TUPLES_HEADER

#include <occa/defines.hpp>
#include <cmath>
#include <iostream>

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
  template <class T>
  class type2 {
  public:
    union { T s0, x; };
    union { T s1, y; };

    inline type2() :
      x(0),
      y(0) {}

    inline type2(const T &x_) :
      x(x_),
      y(0) {}

    inline type2(const T &x_,
                 const T &y_) :
      x(x_),
      y(y_) {}
  };

  template <class T>
  inline type2<T> operator - (const type2<T> &a) {
    return type2<T>(-a.x, -a.y);
  }

  template <class T>
  inline type2<T> operator + (const type2<T> &a, const type2<T> &b) {
    return type2<T>(a.x + b.x,
                     a.y + b.y);
  }

  template <class T>
  inline type2<T> operator + (const T a, const type2<T> &b) {
    return type2<T>(a + b.x,
                     a + b.y);
  }

  template <class T>
  inline type2<T> operator + (const type2<T> &a, const T b) {
    return type2<T>(a.x + b,
                     a.y + b);
  }

  template <class T>
  inline type2<T>& operator += (type2<T> &a, const type2<T> &b) {
    a.x += b.x;
    a.y += b.y;
    return a;
  }

  template <class T>
  inline type2<T>& operator += (type2<T> &a, const T b) {
    a.x += b;
    a.y += b;
    return a;
  }

  template <class T>
  inline type2<T> operator - (const type2<T> &a, const type2<T> &b) {
    return type2<T>(a.x - b.x,
                     a.y - b.y);
  }

  template <class T>
  inline type2<T> operator - (const T a, const type2<T> &b) {
    return type2<T>(a - b.x,
                     a - b.y);
  }

  template <class T>
  inline type2<T> operator - (const type2<T> &a, const T b) {
    return type2<T>(a.x - b,
                     a.y - b);
  }

  template <class T>
  inline type2<T>& operator -= (type2<T> &a, const type2<T> &b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
  }

  template <class T>
  inline type2<T>& operator -= (type2<T> &a, const T b) {
    a.x -= b;
    a.y -= b;
    return a;
  }

  template <class T>
  inline type2<T> operator * (const type2<T> &a, const type2<T> &b) {
    return type2<T>(a.x * b.x,
                     a.y * b.y);
  }

  template <class T>
  inline type2<T> operator * (const T a, const type2<T> &b) {
    return type2<T>(a * b.x,
                     a * b.y);
  }

  template <class T>
  inline type2<T> operator * (const type2<T> &a, const T b) {
    return type2<T>(a.x * b,
                     a.y * b);
  }

  template <class T>
  inline type2<T>& operator *= (type2<T> &a, const type2<T> &b) {
    a.x *= b.x;
    a.y *= b.y;
    return a;
  }

  template <class T>
  inline type2<T>& operator *= (type2<T> &a, const T b) {
    a.x *= b;
    a.y *= b;
    return a;
  }

  template <class T>
  inline type2<T> operator / (const type2<T> &a, const type2<T> &b) {
    return type2<T>(a.x / b.x,
                     a.y / b.y);
  }

  template <class T>
  inline type2<T> operator / (const T a, const type2<T> &b) {
    return type2<T>(a / b.x,
                     a / b.y);
  }

  template <class T>
  inline type2<T> operator / (const type2<T> &a, const T b) {
    return type2<T>(a.x / b,
                     a.y / b);
  }

  template <class T>
  inline type2<T>& operator /= (type2<T> &a, const type2<T> &b) {
    a.x /= b.x;
    a.y /= b.y;
    return a;
  }

  template <class T>
  inline type2<T>& operator /= (type2<T> &a, const T b) {
    a.x /= b;
    a.y /= b;
    return a;
  }

  template <class T>
  inline std::ostream& operator << (std::ostream &out,
                                    const type2<T>& a) {
    out << "[" << a.x << ", " << a.y << "]\n";
    return out;
  }
  //====================================

  //---[ type4 ]------------------------
  template <class T>
  class type4 {
  public:
    union { T s0, x; };
    union { T s1, y; };
    union { T s2, z; };
    union { T s3, w; };

    inline type4() :
      x(0),
      y(0),
      z(0),
      w(0) {}

    inline type4(const T &x_) :
      x(x_),
      y(0),
      z(0),
      w(0) {}

    inline type4(const T &x_,
                 const T &y_) :
      x(x_),
      y(y_),
      z(0),
      w(0) {}

    inline type4(const T &x_,
                 const T &y_,
                 const T &z_) :
      x(x_),
      y(y_),
      z(z_),
      w(0) {}

    inline type4(const T &x_,
                 const T &y_,
                 const T &z_,
                 const T &w_) :
      x(x_),
      y(y_),
      z(z_),
      w(w_) {}
  };

  template <class T>
  inline type4<T> operator - (const type4<T> &a) {
    return type4<T>(-a.x, -a.y, -a.z, -a.w);
  }

  template <class T>
  inline type4<T> operator + (const type4<T> &a, const type4<T> &b) {
    return type4<T>(a.x + b.x,
                     a.y + b.y,
                     a.z + b.z,
                     a.w + b.w);
  }

  template <class T, class T2>
  inline type4<T> operator + (const T2 a, const type4<T> &b) {
    return type4<T>(a + b.x,
                     a + b.y,
                     a + b.z,
                     a + b.w);
  }

  template <class T, class T2>
  inline type4<T> operator + (const type4<T> &a, const T2 b) {
    return type4<T>(a.x + b,
                     a.y + b,
                     a.z + b,
                     a.w + b);
  }

  template <class T>
  inline type4<T>& operator += (type4<T> &a, const type4<T> &b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
  }

  template <class T, class T2>
  inline type4<T>& operator += (type4<T> &a, const T2 b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
    return a;
  }

  template <class T>
  inline type4<T> operator - (const type4<T> &a, const type4<T> &b) {
    return type4<T>(a.x - b.x,
                     a.y - b.y,
                     a.z - b.z,
                     a.w - b.w);
  }

  template <class T, class T2>
  inline type4<T> operator - (const T2 a, const type4<T> &b) {
    return type4<T>(a - b.x,
                     a - b.y,
                     a - b.z,
                     a - b.w);
  }

  template <class T, class T2>
  inline type4<T> operator - (const type4<T> &a, const T2 b) {
    return type4<T>(a.x - b,
                     a.y - b,
                     a.z - b,
                     a.w - b);
  }

  template <class T>
  inline type4<T>& operator -= (type4<T> &a, const type4<T> &b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
  }

  template <class T, class T2>
  inline type4<T>& operator -= (type4<T> &a, const T2 b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
    return a;
  }

  template <class T>
  inline type4<T> operator * (const type4<T> &a, const type4<T> &b) {
    return type4<T>(a.x * b.x,
                     a.y * b.y,
                     a.z * b.z,
                     a.w * b.w);
  }

  template <class T, class T2>
  inline type4<T> operator * (const T2 a, const type4<T> &b) {
    return type4<T>(a * b.x,
                     a * b.y,
                     a * b.z,
                     a * b.w);
  }

  template <class T, class T2>
  inline type4<T> operator * (const type4<T> &a, const T2 b) {
    return type4<T>(a.x * b,
                     a.y * b,
                     a.z * b,
                     a.w * b);
  }

  template <class T>
  inline type4<T>& operator *= (type4<T> &a, const type4<T> &b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    return a;
  }

  template <class T, class T2>
  inline type4<T>& operator *= (type4<T> &a, const T2 b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    return a;
  }

  template <class T>
  inline type4<T> operator / (const type4<T> &a, const type4<T> &b) {
    return type4<T>(a.x / b.x,
                     a.y / b.y,
                     a.z / b.z,
                     a.w / b.w);
  }

  template <class T, class T2>
  inline type4<T> operator / (const T2 a, const type4<T> &b) {
    return type4<T>(a / b.x,
                     a / b.y,
                     a / b.z,
                     a / b.w);
  }

  template <class T, class T2>
  inline type4<T> operator / (const type4<T> &a, const T2 b) {
    return type4<T>(a.x / b,
                     a.y / b,
                     a.z / b,
                     a.w / b);
  }

  template <class T>
  inline type4<T>& operator /= (type4<T> &a, const type4<T> &b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
    return a;
  }

  template <class T, class T2>
  inline type4<T>& operator /= (type4<T> &a, const T2 b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
    return a;
  }

  template <class T>
  inline std::ostream& operator << (std::ostream &out,
                                  const type4<T>& a) {
    out << "[" << a.x << ", " << a.y << ", " << a.z << ", " << a.w << "]\n";
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
  template <class T>
  inline double length(const type2<T> &v) {
    return sqrt((v.x * v.x) +
                (v.y * v.y));
  }

  template <class T>
  inline double length(const type4<T> &v) {
    return sqrt((v.x * v.x) +
                (v.y * v.y) +
                (v.z * v.z) +
                (v.w * v.w));
  }

  template <class T>
  inline type2<T> normalize(const type2<T> &v) {
    const double invNorm = (1.0 / length(v));
    return type2<T>(invNorm * v.x,
                     invNorm * v.y);
  }

  template <class T>
  inline type4<T> normalize(const type4<T> &v) {
    const double invNorm = (1.0 / length(v));
    return type4<T>(invNorm * v.x,
                     invNorm * v.y,
                     invNorm * v.z,
                     invNorm * v.w);
  }

  template <class T>
  inline double dot(const type2<T> &a, const type2<T> &b) {
    return ((a.x * b.x) +
            (a.y * b.y));
  }

  template <class T>
  inline double dot(const type4<T> &a, const type4<T> &b) {
    return ((a.x * b.x) +
            (a.y * b.y) +
            (a.z * b.z) +
            (a.w * b.w));
  }

  template <class T>
  inline T clamp(const T &value, const T min, const T max) {
    return (value < min) ? min : ((max < value) ? max : value);
  }

  template <class T, class T2>
  inline type2<T> clamp(const type2<T> &v, const T2 min, const T2 max) {
    return type2<T>(clamp(v.x, min, max),
                     clamp(v.y, min, max));
  }

  template <class T, class T2>
  inline type4<T> clamp(const type4<T> &v, const T2 min, const T2 max) {
    return type4<T>(clamp(v.x, min, max),
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
