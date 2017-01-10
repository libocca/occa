/* The MIT License (MIT)
 * 
 * Copyright (c) 2014-2016 David Medina and Tim Warburton
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
/*
-------------[ DO NOT EDIT ]-------------
 THIS IS AN AUTOMATICALLY GENERATED FILE
 EDIT: scripts/setupVectorDefines.py
=========================================
*/
#if (!defined(OCCA_IN_KERNEL) || (!OCCA_USING_OPENCL))
#  include <iostream>
#  include <cmath>
#  include "occa/defines.hpp"

#  ifndef OCCA_IN_KERNEL
#  define occaFunction
namespace occa {
#  endif

//---[ bool2 ]--------------------------
#define OCCA_BOOL2 bool2
class bool2{
public:
  union { bool s0, x; };
  union { bool s1, y; };

  inline occaFunction bool2() : 
    x(0),
    y(0) {}

  inline occaFunction bool2(const bool &x_) : 
    x(x_),
    y(0) {}

  inline occaFunction bool2(const bool &x_,
                            const bool &y_) : 
    x(x_),
    y(y_) {}
};

occaFunction inline bool2  operator +  (const bool2 &a, const bool2 &b) {
  return OCCA_BOOL2(a.x + b.x,
                    a.y + b.y);
}

occaFunction inline bool2  operator +  (const bool &a, const bool2 &b) {
  return OCCA_BOOL2(a + b.x,
                    a + b.y);
}

occaFunction inline bool2  operator +  (const bool2 &a, const bool &b) {
  return OCCA_BOOL2(a.x + b,
                    a.y + b);
}

occaFunction inline bool2& operator += (      bool2 &a, const bool2 &b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}

occaFunction inline bool2& operator += (      bool2 &a, const bool &b) {
  a.x += b;
  a.y += b;
  return a;
}
occaFunction inline bool2  operator -  (const bool2 &a, const bool2 &b) {
  return OCCA_BOOL2(a.x - b.x,
                    a.y - b.y);
}

occaFunction inline bool2  operator -  (const bool &a, const bool2 &b) {
  return OCCA_BOOL2(a - b.x,
                    a - b.y);
}

occaFunction inline bool2  operator -  (const bool2 &a, const bool &b) {
  return OCCA_BOOL2(a.x - b,
                    a.y - b);
}

occaFunction inline bool2& operator -= (      bool2 &a, const bool2 &b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

occaFunction inline bool2& operator -= (      bool2 &a, const bool &b) {
  a.x -= b;
  a.y -= b;
  return a;
}
occaFunction inline bool2  operator *  (const bool2 &a, const bool2 &b) {
  return OCCA_BOOL2(a.x * b.x,
                    a.y * b.y);
}

occaFunction inline bool2  operator *  (const bool &a, const bool2 &b) {
  return OCCA_BOOL2(a * b.x,
                    a * b.y);
}

occaFunction inline bool2  operator *  (const bool2 &a, const bool &b) {
  return OCCA_BOOL2(a.x * b,
                    a.y * b);
}

occaFunction inline bool2& operator *= (      bool2 &a, const bool2 &b) {
  a.x *= b.x;
  a.y *= b.y;
  return a;
}

occaFunction inline bool2& operator *= (      bool2 &a, const bool &b) {
  a.x *= b;
  a.y *= b;
  return a;
}
occaFunction inline bool2  operator /  (const bool2 &a, const bool2 &b) {
  return OCCA_BOOL2(a.x / b.x,
                    a.y / b.y);
}

occaFunction inline bool2  operator /  (const bool &a, const bool2 &b) {
  return OCCA_BOOL2(a / b.x,
                    a / b.y);
}

occaFunction inline bool2  operator /  (const bool2 &a, const bool &b) {
  return OCCA_BOOL2(a.x / b,
                    a.y / b);
}

occaFunction inline bool2& operator /= (      bool2 &a, const bool2 &b) {
  a.x /= b.x;
  a.y /= b.y;
  return a;
}

occaFunction inline bool2& operator /= (      bool2 &a, const bool &b) {
  a.x /= b;
  a.y /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const bool2& a) {
  out << "[" << (a.x ? "true" : "false") << ", "
             << (a.y ? "true" : "false")
      << "]\n";

  return out;
}
#endif

//======================================


//---[ bool4 ]--------------------------
#define OCCA_BOOL4 bool4
class bool4{
public:
  union { bool s0, x; };
  union { bool s1, y; };
  union { bool s2, z; };
  union { bool s3, w; };

  inline occaFunction bool4() : 
    x(0),
    y(0),
    z(0),
    w(0) {}

  inline occaFunction bool4(const bool &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0) {}

  inline occaFunction bool4(const bool &x_,
                            const bool &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0) {}

  inline occaFunction bool4(const bool &x_,
                            const bool &y_,
                            const bool &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0) {}

  inline occaFunction bool4(const bool &x_,
                            const bool &y_,
                            const bool &z_,
                            const bool &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_) {}
};

occaFunction inline bool4  operator +  (const bool4 &a, const bool4 &b) {
  return OCCA_BOOL4(a.x + b.x,
                    a.y + b.y,
                    a.z + b.z,
                    a.w + b.w);
}

occaFunction inline bool4  operator +  (const bool &a, const bool4 &b) {
  return OCCA_BOOL4(a + b.x,
                    a + b.y,
                    a + b.z,
                    a + b.w);
}

occaFunction inline bool4  operator +  (const bool4 &a, const bool &b) {
  return OCCA_BOOL4(a.x + b,
                    a.y + b,
                    a.z + b,
                    a.w + b);
}

occaFunction inline bool4& operator += (      bool4 &a, const bool4 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

occaFunction inline bool4& operator += (      bool4 &a, const bool &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  return a;
}
occaFunction inline bool4  operator -  (const bool4 &a, const bool4 &b) {
  return OCCA_BOOL4(a.x - b.x,
                    a.y - b.y,
                    a.z - b.z,
                    a.w - b.w);
}

occaFunction inline bool4  operator -  (const bool &a, const bool4 &b) {
  return OCCA_BOOL4(a - b.x,
                    a - b.y,
                    a - b.z,
                    a - b.w);
}

occaFunction inline bool4  operator -  (const bool4 &a, const bool &b) {
  return OCCA_BOOL4(a.x - b,
                    a.y - b,
                    a.z - b,
                    a.w - b);
}

occaFunction inline bool4& operator -= (      bool4 &a, const bool4 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  return a;
}

occaFunction inline bool4& operator -= (      bool4 &a, const bool &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  return a;
}
occaFunction inline bool4  operator *  (const bool4 &a, const bool4 &b) {
  return OCCA_BOOL4(a.x * b.x,
                    a.y * b.y,
                    a.z * b.z,
                    a.w * b.w);
}

occaFunction inline bool4  operator *  (const bool &a, const bool4 &b) {
  return OCCA_BOOL4(a * b.x,
                    a * b.y,
                    a * b.z,
                    a * b.w);
}

occaFunction inline bool4  operator *  (const bool4 &a, const bool &b) {
  return OCCA_BOOL4(a.x * b,
                    a.y * b,
                    a.z * b,
                    a.w * b);
}

occaFunction inline bool4& operator *= (      bool4 &a, const bool4 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  return a;
}

occaFunction inline bool4& operator *= (      bool4 &a, const bool &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}
occaFunction inline bool4  operator /  (const bool4 &a, const bool4 &b) {
  return OCCA_BOOL4(a.x / b.x,
                    a.y / b.y,
                    a.z / b.z,
                    a.w / b.w);
}

occaFunction inline bool4  operator /  (const bool &a, const bool4 &b) {
  return OCCA_BOOL4(a / b.x,
                    a / b.y,
                    a / b.z,
                    a / b.w);
}

occaFunction inline bool4  operator /  (const bool4 &a, const bool &b) {
  return OCCA_BOOL4(a.x / b,
                    a.y / b,
                    a.z / b,
                    a.w / b);
}

occaFunction inline bool4& operator /= (      bool4 &a, const bool4 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  return a;
}

occaFunction inline bool4& operator /= (      bool4 &a, const bool &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const bool4& a) {
  out << "[" << (a.x ? "true" : "false") << ", "
             << (a.y ? "true" : "false") << ", "
             << (a.z ? "true" : "false") << ", "
             << (a.w ? "true" : "false")
      << "]\n";

  return out;
}
#endif

//======================================


//---[ bool3 ]--------------------------
#define OCCA_BOOL3 bool3
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
typedef bool4 bool3;
#endif
//======================================


//---[ bool8 ]--------------------------
#define OCCA_BOOL8 bool8
class bool8{
public:
  union { bool s0, x; };
  union { bool s1, y; };
  union { bool s2, z; };
  union { bool s3, w; };
  bool s4;
  bool s5;
  bool s6;
  bool s7;

  inline occaFunction bool8() : 
    x(0),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction bool8(const bool &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction bool8(const bool &x_,
                            const bool &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction bool8(const bool &x_,
                            const bool &y_,
                            const bool &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction bool8(const bool &x_,
                            const bool &y_,
                            const bool &z_,
                            const bool &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction bool8(const bool &x_,
                            const bool &y_,
                            const bool &z_,
                            const bool &w_,
                            const bool &s4_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction bool8(const bool &x_,
                            const bool &y_,
                            const bool &z_,
                            const bool &w_,
                            const bool &s4_,
                            const bool &s5_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(0),
    s7(0) {}

  inline occaFunction bool8(const bool &x_,
                            const bool &y_,
                            const bool &z_,
                            const bool &w_,
                            const bool &s4_,
                            const bool &s5_,
                            const bool &s6_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(0) {}

  inline occaFunction bool8(const bool &x_,
                            const bool &y_,
                            const bool &z_,
                            const bool &w_,
                            const bool &s4_,
                            const bool &s5_,
                            const bool &s6_,
                            const bool &s7_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_) {}
};

occaFunction inline bool8  operator +  (const bool8 &a, const bool8 &b) {
  return OCCA_BOOL8(a.x + b.x,
                    a.y + b.y,
                    a.z + b.z,
                    a.w + b.w,
                    a.s4 + b.s4,
                    a.s5 + b.s5,
                    a.s6 + b.s6,
                    a.s7 + b.s7);
}

occaFunction inline bool8  operator +  (const bool &a, const bool8 &b) {
  return OCCA_BOOL8(a + b.x,
                    a + b.y,
                    a + b.z,
                    a + b.w,
                    a + b.s4,
                    a + b.s5,
                    a + b.s6,
                    a + b.s7);
}

occaFunction inline bool8  operator +  (const bool8 &a, const bool &b) {
  return OCCA_BOOL8(a.x + b,
                    a.y + b,
                    a.z + b,
                    a.w + b,
                    a.s4 + b,
                    a.s5 + b,
                    a.s6 + b,
                    a.s7 + b);
}

occaFunction inline bool8& operator += (      bool8 &a, const bool8 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  a.s4 += b.s4;
  a.s5 += b.s5;
  a.s6 += b.s6;
  a.s7 += b.s7;
  return a;
}

occaFunction inline bool8& operator += (      bool8 &a, const bool &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  a.s4 += b;
  a.s5 += b;
  a.s6 += b;
  a.s7 += b;
  return a;
}
occaFunction inline bool8  operator -  (const bool8 &a, const bool8 &b) {
  return OCCA_BOOL8(a.x - b.x,
                    a.y - b.y,
                    a.z - b.z,
                    a.w - b.w,
                    a.s4 - b.s4,
                    a.s5 - b.s5,
                    a.s6 - b.s6,
                    a.s7 - b.s7);
}

occaFunction inline bool8  operator -  (const bool &a, const bool8 &b) {
  return OCCA_BOOL8(a - b.x,
                    a - b.y,
                    a - b.z,
                    a - b.w,
                    a - b.s4,
                    a - b.s5,
                    a - b.s6,
                    a - b.s7);
}

occaFunction inline bool8  operator -  (const bool8 &a, const bool &b) {
  return OCCA_BOOL8(a.x - b,
                    a.y - b,
                    a.z - b,
                    a.w - b,
                    a.s4 - b,
                    a.s5 - b,
                    a.s6 - b,
                    a.s7 - b);
}

occaFunction inline bool8& operator -= (      bool8 &a, const bool8 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  a.s4 -= b.s4;
  a.s5 -= b.s5;
  a.s6 -= b.s6;
  a.s7 -= b.s7;
  return a;
}

occaFunction inline bool8& operator -= (      bool8 &a, const bool &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  a.s4 -= b;
  a.s5 -= b;
  a.s6 -= b;
  a.s7 -= b;
  return a;
}
occaFunction inline bool8  operator *  (const bool8 &a, const bool8 &b) {
  return OCCA_BOOL8(a.x * b.x,
                    a.y * b.y,
                    a.z * b.z,
                    a.w * b.w,
                    a.s4 * b.s4,
                    a.s5 * b.s5,
                    a.s6 * b.s6,
                    a.s7 * b.s7);
}

occaFunction inline bool8  operator *  (const bool &a, const bool8 &b) {
  return OCCA_BOOL8(a * b.x,
                    a * b.y,
                    a * b.z,
                    a * b.w,
                    a * b.s4,
                    a * b.s5,
                    a * b.s6,
                    a * b.s7);
}

occaFunction inline bool8  operator *  (const bool8 &a, const bool &b) {
  return OCCA_BOOL8(a.x * b,
                    a.y * b,
                    a.z * b,
                    a.w * b,
                    a.s4 * b,
                    a.s5 * b,
                    a.s6 * b,
                    a.s7 * b);
}

occaFunction inline bool8& operator *= (      bool8 &a, const bool8 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  a.s4 *= b.s4;
  a.s5 *= b.s5;
  a.s6 *= b.s6;
  a.s7 *= b.s7;
  return a;
}

occaFunction inline bool8& operator *= (      bool8 &a, const bool &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  a.s4 *= b;
  a.s5 *= b;
  a.s6 *= b;
  a.s7 *= b;
  return a;
}
occaFunction inline bool8  operator /  (const bool8 &a, const bool8 &b) {
  return OCCA_BOOL8(a.x / b.x,
                    a.y / b.y,
                    a.z / b.z,
                    a.w / b.w,
                    a.s4 / b.s4,
                    a.s5 / b.s5,
                    a.s6 / b.s6,
                    a.s7 / b.s7);
}

occaFunction inline bool8  operator /  (const bool &a, const bool8 &b) {
  return OCCA_BOOL8(a / b.x,
                    a / b.y,
                    a / b.z,
                    a / b.w,
                    a / b.s4,
                    a / b.s5,
                    a / b.s6,
                    a / b.s7);
}

occaFunction inline bool8  operator /  (const bool8 &a, const bool &b) {
  return OCCA_BOOL8(a.x / b,
                    a.y / b,
                    a.z / b,
                    a.w / b,
                    a.s4 / b,
                    a.s5 / b,
                    a.s6 / b,
                    a.s7 / b);
}

occaFunction inline bool8& operator /= (      bool8 &a, const bool8 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  a.s4 /= b.s4;
  a.s5 /= b.s5;
  a.s6 /= b.s6;
  a.s7 /= b.s7;
  return a;
}

occaFunction inline bool8& operator /= (      bool8 &a, const bool &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  a.s4 /= b;
  a.s5 /= b;
  a.s6 /= b;
  a.s7 /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const bool8& a) {
  out << "[" << (a.x ? "true" : "false") << ", "
             << (a.y ? "true" : "false") << ", "
             << (a.z ? "true" : "false") << ", "
             << (a.w ? "true" : "false") << ", "
             << (a.s4 ? "true" : "false") << ", "
             << (a.s5 ? "true" : "false") << ", "
             << (a.s6 ? "true" : "false") << ", "
             << (a.s7 ? "true" : "false")
      << "]\n";

  return out;
}
#endif

//======================================


//---[ bool16 ]-------------------------
#define OCCA_BOOL16 bool16
class bool16{
public:
  union { bool s0, x; };
  union { bool s1, y; };
  union { bool s2, z; };
  union { bool s3, w; };
  bool s4;
  bool s5;
  bool s6;
  bool s7;
  bool s8;
  bool s9;
  bool s10;
  bool s11;
  bool s12;
  bool s13;
  bool s14;
  bool s15;

  inline occaFunction bool16() : 
    x(0),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction bool16(const bool &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction bool16(const bool &x_,
                             const bool &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction bool16(const bool &x_,
                             const bool &y_,
                             const bool &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction bool16(const bool &x_,
                             const bool &y_,
                             const bool &z_,
                             const bool &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction bool16(const bool &x_,
                             const bool &y_,
                             const bool &z_,
                             const bool &w_,
                             const bool &s4_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction bool16(const bool &x_,
                             const bool &y_,
                             const bool &z_,
                             const bool &w_,
                             const bool &s4_,
                             const bool &s5_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction bool16(const bool &x_,
                             const bool &y_,
                             const bool &z_,
                             const bool &w_,
                             const bool &s4_,
                             const bool &s5_,
                             const bool &s6_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction bool16(const bool &x_,
                             const bool &y_,
                             const bool &z_,
                             const bool &w_,
                             const bool &s4_,
                             const bool &s5_,
                             const bool &s6_,
                             const bool &s7_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction bool16(const bool &x_,
                             const bool &y_,
                             const bool &z_,
                             const bool &w_,
                             const bool &s4_,
                             const bool &s5_,
                             const bool &s6_,
                             const bool &s7_,
                             const bool &s8_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction bool16(const bool &x_,
                             const bool &y_,
                             const bool &z_,
                             const bool &w_,
                             const bool &s4_,
                             const bool &s5_,
                             const bool &s6_,
                             const bool &s7_,
                             const bool &s8_,
                             const bool &s9_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction bool16(const bool &x_,
                             const bool &y_,
                             const bool &z_,
                             const bool &w_,
                             const bool &s4_,
                             const bool &s5_,
                             const bool &s6_,
                             const bool &s7_,
                             const bool &s8_,
                             const bool &s9_,
                             const bool &s10_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction bool16(const bool &x_,
                             const bool &y_,
                             const bool &z_,
                             const bool &w_,
                             const bool &s4_,
                             const bool &s5_,
                             const bool &s6_,
                             const bool &s7_,
                             const bool &s8_,
                             const bool &s9_,
                             const bool &s10_,
                             const bool &s11_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction bool16(const bool &x_,
                             const bool &y_,
                             const bool &z_,
                             const bool &w_,
                             const bool &s4_,
                             const bool &s5_,
                             const bool &s6_,
                             const bool &s7_,
                             const bool &s8_,
                             const bool &s9_,
                             const bool &s10_,
                             const bool &s11_,
                             const bool &s12_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction bool16(const bool &x_,
                             const bool &y_,
                             const bool &z_,
                             const bool &w_,
                             const bool &s4_,
                             const bool &s5_,
                             const bool &s6_,
                             const bool &s7_,
                             const bool &s8_,
                             const bool &s9_,
                             const bool &s10_,
                             const bool &s11_,
                             const bool &s12_,
                             const bool &s13_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(0),
    s15(0) {}

  inline occaFunction bool16(const bool &x_,
                             const bool &y_,
                             const bool &z_,
                             const bool &w_,
                             const bool &s4_,
                             const bool &s5_,
                             const bool &s6_,
                             const bool &s7_,
                             const bool &s8_,
                             const bool &s9_,
                             const bool &s10_,
                             const bool &s11_,
                             const bool &s12_,
                             const bool &s13_,
                             const bool &s14_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(s14_),
    s15(0) {}

  inline occaFunction bool16(const bool &x_,
                             const bool &y_,
                             const bool &z_,
                             const bool &w_,
                             const bool &s4_,
                             const bool &s5_,
                             const bool &s6_,
                             const bool &s7_,
                             const bool &s8_,
                             const bool &s9_,
                             const bool &s10_,
                             const bool &s11_,
                             const bool &s12_,
                             const bool &s13_,
                             const bool &s14_,
                             const bool &s15_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(s14_),
    s15(s15_) {}
};

occaFunction inline bool16  operator +  (const bool16 &a, const bool16 &b) {
  return OCCA_BOOL16(a.x + b.x,
                     a.y + b.y,
                     a.z + b.z,
                     a.w + b.w,
                     a.s4 + b.s4,
                     a.s5 + b.s5,
                     a.s6 + b.s6,
                     a.s7 + b.s7,
                     a.s8 + b.s8,
                     a.s9 + b.s9,
                     a.s10 + b.s10,
                     a.s11 + b.s11,
                     a.s12 + b.s12,
                     a.s13 + b.s13,
                     a.s14 + b.s14,
                     a.s15 + b.s15);
}

occaFunction inline bool16  operator +  (const bool &a, const bool16 &b) {
  return OCCA_BOOL16(a + b.x,
                     a + b.y,
                     a + b.z,
                     a + b.w,
                     a + b.s4,
                     a + b.s5,
                     a + b.s6,
                     a + b.s7,
                     a + b.s8,
                     a + b.s9,
                     a + b.s10,
                     a + b.s11,
                     a + b.s12,
                     a + b.s13,
                     a + b.s14,
                     a + b.s15);
}

occaFunction inline bool16  operator +  (const bool16 &a, const bool &b) {
  return OCCA_BOOL16(a.x + b,
                     a.y + b,
                     a.z + b,
                     a.w + b,
                     a.s4 + b,
                     a.s5 + b,
                     a.s6 + b,
                     a.s7 + b,
                     a.s8 + b,
                     a.s9 + b,
                     a.s10 + b,
                     a.s11 + b,
                     a.s12 + b,
                     a.s13 + b,
                     a.s14 + b,
                     a.s15 + b);
}

occaFunction inline bool16& operator += (      bool16 &a, const bool16 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  a.s4 += b.s4;
  a.s5 += b.s5;
  a.s6 += b.s6;
  a.s7 += b.s7;
  a.s8 += b.s8;
  a.s9 += b.s9;
  a.s10 += b.s10;
  a.s11 += b.s11;
  a.s12 += b.s12;
  a.s13 += b.s13;
  a.s14 += b.s14;
  a.s15 += b.s15;
  return a;
}

occaFunction inline bool16& operator += (      bool16 &a, const bool &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  a.s4 += b;
  a.s5 += b;
  a.s6 += b;
  a.s7 += b;
  a.s8 += b;
  a.s9 += b;
  a.s10 += b;
  a.s11 += b;
  a.s12 += b;
  a.s13 += b;
  a.s14 += b;
  a.s15 += b;
  return a;
}
occaFunction inline bool16  operator -  (const bool16 &a, const bool16 &b) {
  return OCCA_BOOL16(a.x - b.x,
                     a.y - b.y,
                     a.z - b.z,
                     a.w - b.w,
                     a.s4 - b.s4,
                     a.s5 - b.s5,
                     a.s6 - b.s6,
                     a.s7 - b.s7,
                     a.s8 - b.s8,
                     a.s9 - b.s9,
                     a.s10 - b.s10,
                     a.s11 - b.s11,
                     a.s12 - b.s12,
                     a.s13 - b.s13,
                     a.s14 - b.s14,
                     a.s15 - b.s15);
}

occaFunction inline bool16  operator -  (const bool &a, const bool16 &b) {
  return OCCA_BOOL16(a - b.x,
                     a - b.y,
                     a - b.z,
                     a - b.w,
                     a - b.s4,
                     a - b.s5,
                     a - b.s6,
                     a - b.s7,
                     a - b.s8,
                     a - b.s9,
                     a - b.s10,
                     a - b.s11,
                     a - b.s12,
                     a - b.s13,
                     a - b.s14,
                     a - b.s15);
}

occaFunction inline bool16  operator -  (const bool16 &a, const bool &b) {
  return OCCA_BOOL16(a.x - b,
                     a.y - b,
                     a.z - b,
                     a.w - b,
                     a.s4 - b,
                     a.s5 - b,
                     a.s6 - b,
                     a.s7 - b,
                     a.s8 - b,
                     a.s9 - b,
                     a.s10 - b,
                     a.s11 - b,
                     a.s12 - b,
                     a.s13 - b,
                     a.s14 - b,
                     a.s15 - b);
}

occaFunction inline bool16& operator -= (      bool16 &a, const bool16 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  a.s4 -= b.s4;
  a.s5 -= b.s5;
  a.s6 -= b.s6;
  a.s7 -= b.s7;
  a.s8 -= b.s8;
  a.s9 -= b.s9;
  a.s10 -= b.s10;
  a.s11 -= b.s11;
  a.s12 -= b.s12;
  a.s13 -= b.s13;
  a.s14 -= b.s14;
  a.s15 -= b.s15;
  return a;
}

occaFunction inline bool16& operator -= (      bool16 &a, const bool &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  a.s4 -= b;
  a.s5 -= b;
  a.s6 -= b;
  a.s7 -= b;
  a.s8 -= b;
  a.s9 -= b;
  a.s10 -= b;
  a.s11 -= b;
  a.s12 -= b;
  a.s13 -= b;
  a.s14 -= b;
  a.s15 -= b;
  return a;
}
occaFunction inline bool16  operator *  (const bool16 &a, const bool16 &b) {
  return OCCA_BOOL16(a.x * b.x,
                     a.y * b.y,
                     a.z * b.z,
                     a.w * b.w,
                     a.s4 * b.s4,
                     a.s5 * b.s5,
                     a.s6 * b.s6,
                     a.s7 * b.s7,
                     a.s8 * b.s8,
                     a.s9 * b.s9,
                     a.s10 * b.s10,
                     a.s11 * b.s11,
                     a.s12 * b.s12,
                     a.s13 * b.s13,
                     a.s14 * b.s14,
                     a.s15 * b.s15);
}

occaFunction inline bool16  operator *  (const bool &a, const bool16 &b) {
  return OCCA_BOOL16(a * b.x,
                     a * b.y,
                     a * b.z,
                     a * b.w,
                     a * b.s4,
                     a * b.s5,
                     a * b.s6,
                     a * b.s7,
                     a * b.s8,
                     a * b.s9,
                     a * b.s10,
                     a * b.s11,
                     a * b.s12,
                     a * b.s13,
                     a * b.s14,
                     a * b.s15);
}

occaFunction inline bool16  operator *  (const bool16 &a, const bool &b) {
  return OCCA_BOOL16(a.x * b,
                     a.y * b,
                     a.z * b,
                     a.w * b,
                     a.s4 * b,
                     a.s5 * b,
                     a.s6 * b,
                     a.s7 * b,
                     a.s8 * b,
                     a.s9 * b,
                     a.s10 * b,
                     a.s11 * b,
                     a.s12 * b,
                     a.s13 * b,
                     a.s14 * b,
                     a.s15 * b);
}

occaFunction inline bool16& operator *= (      bool16 &a, const bool16 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  a.s4 *= b.s4;
  a.s5 *= b.s5;
  a.s6 *= b.s6;
  a.s7 *= b.s7;
  a.s8 *= b.s8;
  a.s9 *= b.s9;
  a.s10 *= b.s10;
  a.s11 *= b.s11;
  a.s12 *= b.s12;
  a.s13 *= b.s13;
  a.s14 *= b.s14;
  a.s15 *= b.s15;
  return a;
}

occaFunction inline bool16& operator *= (      bool16 &a, const bool &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  a.s4 *= b;
  a.s5 *= b;
  a.s6 *= b;
  a.s7 *= b;
  a.s8 *= b;
  a.s9 *= b;
  a.s10 *= b;
  a.s11 *= b;
  a.s12 *= b;
  a.s13 *= b;
  a.s14 *= b;
  a.s15 *= b;
  return a;
}
occaFunction inline bool16  operator /  (const bool16 &a, const bool16 &b) {
  return OCCA_BOOL16(a.x / b.x,
                     a.y / b.y,
                     a.z / b.z,
                     a.w / b.w,
                     a.s4 / b.s4,
                     a.s5 / b.s5,
                     a.s6 / b.s6,
                     a.s7 / b.s7,
                     a.s8 / b.s8,
                     a.s9 / b.s9,
                     a.s10 / b.s10,
                     a.s11 / b.s11,
                     a.s12 / b.s12,
                     a.s13 / b.s13,
                     a.s14 / b.s14,
                     a.s15 / b.s15);
}

occaFunction inline bool16  operator /  (const bool &a, const bool16 &b) {
  return OCCA_BOOL16(a / b.x,
                     a / b.y,
                     a / b.z,
                     a / b.w,
                     a / b.s4,
                     a / b.s5,
                     a / b.s6,
                     a / b.s7,
                     a / b.s8,
                     a / b.s9,
                     a / b.s10,
                     a / b.s11,
                     a / b.s12,
                     a / b.s13,
                     a / b.s14,
                     a / b.s15);
}

occaFunction inline bool16  operator /  (const bool16 &a, const bool &b) {
  return OCCA_BOOL16(a.x / b,
                     a.y / b,
                     a.z / b,
                     a.w / b,
                     a.s4 / b,
                     a.s5 / b,
                     a.s6 / b,
                     a.s7 / b,
                     a.s8 / b,
                     a.s9 / b,
                     a.s10 / b,
                     a.s11 / b,
                     a.s12 / b,
                     a.s13 / b,
                     a.s14 / b,
                     a.s15 / b);
}

occaFunction inline bool16& operator /= (      bool16 &a, const bool16 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  a.s4 /= b.s4;
  a.s5 /= b.s5;
  a.s6 /= b.s6;
  a.s7 /= b.s7;
  a.s8 /= b.s8;
  a.s9 /= b.s9;
  a.s10 /= b.s10;
  a.s11 /= b.s11;
  a.s12 /= b.s12;
  a.s13 /= b.s13;
  a.s14 /= b.s14;
  a.s15 /= b.s15;
  return a;
}

occaFunction inline bool16& operator /= (      bool16 &a, const bool &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  a.s4 /= b;
  a.s5 /= b;
  a.s6 /= b;
  a.s7 /= b;
  a.s8 /= b;
  a.s9 /= b;
  a.s10 /= b;
  a.s11 /= b;
  a.s12 /= b;
  a.s13 /= b;
  a.s14 /= b;
  a.s15 /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const bool16& a) {
  out << "[" << (a.x ? "true" : "false") << ", "
             << (a.y ? "true" : "false") << ", "
             << (a.z ? "true" : "false") << ", "
             << (a.w ? "true" : "false") << ", "
             << (a.s4 ? "true" : "false") << ", "
             << (a.s5 ? "true" : "false") << ", "
             << (a.s6 ? "true" : "false") << ", "
             << (a.s7 ? "true" : "false") << ", "
             << (a.s8 ? "true" : "false") << ", "
             << (a.s9 ? "true" : "false") << ", "
             << (a.s10 ? "true" : "false") << ", "
             << (a.s11 ? "true" : "false") << ", "
             << (a.s12 ? "true" : "false") << ", "
             << (a.s13 ? "true" : "false") << ", "
             << (a.s14 ? "true" : "false") << ", "
             << (a.s15 ? "true" : "false")
      << "]\n";

  return out;
}
#endif

//======================================


//---[ char2 ]--------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_CHAR2 make_char2
#else
#  define OCCA_CHAR2 char2
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
class char2{
public:
  union { char s0, x; };
  union { char s1, y; };

  inline occaFunction char2() : 
    x(0),
    y(0) {}

  inline occaFunction char2(const char &x_) : 
    x(x_),
    y(0) {}

  inline occaFunction char2(const char &x_,
                            const char &y_) : 
    x(x_),
    y(y_) {}
};
#endif

occaFunction inline char2 operator + (const char2 &a) {
  return OCCA_CHAR2(+a.x,
                    +a.y);
}

occaFunction inline char2 operator ++ (char2 &a, int) {
  return OCCA_CHAR2(a.x++,
                    a.y++);
}

occaFunction inline char2& operator ++ (char2 &a) {
  ++a.x;
  ++a.y;
  return a;
}
occaFunction inline char2 operator - (const char2 &a) {
  return OCCA_CHAR2(-a.x,
                    -a.y);
}

occaFunction inline char2 operator -- (char2 &a, int) {
  return OCCA_CHAR2(a.x--,
                    a.y--);
}

occaFunction inline char2& operator -- (char2 &a) {
  --a.x;
  --a.y;
  return a;
}
occaFunction inline char2  operator +  (const char2 &a, const char2 &b) {
  return OCCA_CHAR2(a.x + b.x,
                    a.y + b.y);
}

occaFunction inline char2  operator +  (const char &a, const char2 &b) {
  return OCCA_CHAR2(a + b.x,
                    a + b.y);
}

occaFunction inline char2  operator +  (const char2 &a, const char &b) {
  return OCCA_CHAR2(a.x + b,
                    a.y + b);
}

occaFunction inline char2& operator += (      char2 &a, const char2 &b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}

occaFunction inline char2& operator += (      char2 &a, const char &b) {
  a.x += b;
  a.y += b;
  return a;
}
occaFunction inline char2  operator -  (const char2 &a, const char2 &b) {
  return OCCA_CHAR2(a.x - b.x,
                    a.y - b.y);
}

occaFunction inline char2  operator -  (const char &a, const char2 &b) {
  return OCCA_CHAR2(a - b.x,
                    a - b.y);
}

occaFunction inline char2  operator -  (const char2 &a, const char &b) {
  return OCCA_CHAR2(a.x - b,
                    a.y - b);
}

occaFunction inline char2& operator -= (      char2 &a, const char2 &b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

occaFunction inline char2& operator -= (      char2 &a, const char &b) {
  a.x -= b;
  a.y -= b;
  return a;
}
occaFunction inline char2  operator *  (const char2 &a, const char2 &b) {
  return OCCA_CHAR2(a.x * b.x,
                    a.y * b.y);
}

occaFunction inline char2  operator *  (const char &a, const char2 &b) {
  return OCCA_CHAR2(a * b.x,
                    a * b.y);
}

occaFunction inline char2  operator *  (const char2 &a, const char &b) {
  return OCCA_CHAR2(a.x * b,
                    a.y * b);
}

occaFunction inline char2& operator *= (      char2 &a, const char2 &b) {
  a.x *= b.x;
  a.y *= b.y;
  return a;
}

occaFunction inline char2& operator *= (      char2 &a, const char &b) {
  a.x *= b;
  a.y *= b;
  return a;
}
occaFunction inline char2  operator /  (const char2 &a, const char2 &b) {
  return OCCA_CHAR2(a.x / b.x,
                    a.y / b.y);
}

occaFunction inline char2  operator /  (const char &a, const char2 &b) {
  return OCCA_CHAR2(a / b.x,
                    a / b.y);
}

occaFunction inline char2  operator /  (const char2 &a, const char &b) {
  return OCCA_CHAR2(a.x / b,
                    a.y / b);
}

occaFunction inline char2& operator /= (      char2 &a, const char2 &b) {
  a.x /= b.x;
  a.y /= b.y;
  return a;
}

occaFunction inline char2& operator /= (      char2 &a, const char &b) {
  a.x /= b;
  a.y /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const char2& a) {
  out << "[" << a.x << ", "
             << a.y
      << "]\n";

  return out;
}
#endif

//======================================


//---[ char4 ]--------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_CHAR4 make_char4
#else
#  define OCCA_CHAR4 char4
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
class char4{
public:
  union { char s0, x; };
  union { char s1, y; };
  union { char s2, z; };
  union { char s3, w; };

  inline occaFunction char4() : 
    x(0),
    y(0),
    z(0),
    w(0) {}

  inline occaFunction char4(const char &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0) {}

  inline occaFunction char4(const char &x_,
                            const char &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0) {}

  inline occaFunction char4(const char &x_,
                            const char &y_,
                            const char &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0) {}

  inline occaFunction char4(const char &x_,
                            const char &y_,
                            const char &z_,
                            const char &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_) {}
};
#endif

occaFunction inline char4 operator + (const char4 &a) {
  return OCCA_CHAR4(+a.x,
                    +a.y,
                    +a.z,
                    +a.w);
}

occaFunction inline char4 operator ++ (char4 &a, int) {
  return OCCA_CHAR4(a.x++,
                    a.y++,
                    a.z++,
                    a.w++);
}

occaFunction inline char4& operator ++ (char4 &a) {
  ++a.x;
  ++a.y;
  ++a.z;
  ++a.w;
  return a;
}
occaFunction inline char4 operator - (const char4 &a) {
  return OCCA_CHAR4(-a.x,
                    -a.y,
                    -a.z,
                    -a.w);
}

occaFunction inline char4 operator -- (char4 &a, int) {
  return OCCA_CHAR4(a.x--,
                    a.y--,
                    a.z--,
                    a.w--);
}

occaFunction inline char4& operator -- (char4 &a) {
  --a.x;
  --a.y;
  --a.z;
  --a.w;
  return a;
}
occaFunction inline char4  operator +  (const char4 &a, const char4 &b) {
  return OCCA_CHAR4(a.x + b.x,
                    a.y + b.y,
                    a.z + b.z,
                    a.w + b.w);
}

occaFunction inline char4  operator +  (const char &a, const char4 &b) {
  return OCCA_CHAR4(a + b.x,
                    a + b.y,
                    a + b.z,
                    a + b.w);
}

occaFunction inline char4  operator +  (const char4 &a, const char &b) {
  return OCCA_CHAR4(a.x + b,
                    a.y + b,
                    a.z + b,
                    a.w + b);
}

occaFunction inline char4& operator += (      char4 &a, const char4 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

occaFunction inline char4& operator += (      char4 &a, const char &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  return a;
}
occaFunction inline char4  operator -  (const char4 &a, const char4 &b) {
  return OCCA_CHAR4(a.x - b.x,
                    a.y - b.y,
                    a.z - b.z,
                    a.w - b.w);
}

occaFunction inline char4  operator -  (const char &a, const char4 &b) {
  return OCCA_CHAR4(a - b.x,
                    a - b.y,
                    a - b.z,
                    a - b.w);
}

occaFunction inline char4  operator -  (const char4 &a, const char &b) {
  return OCCA_CHAR4(a.x - b,
                    a.y - b,
                    a.z - b,
                    a.w - b);
}

occaFunction inline char4& operator -= (      char4 &a, const char4 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  return a;
}

occaFunction inline char4& operator -= (      char4 &a, const char &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  return a;
}
occaFunction inline char4  operator *  (const char4 &a, const char4 &b) {
  return OCCA_CHAR4(a.x * b.x,
                    a.y * b.y,
                    a.z * b.z,
                    a.w * b.w);
}

occaFunction inline char4  operator *  (const char &a, const char4 &b) {
  return OCCA_CHAR4(a * b.x,
                    a * b.y,
                    a * b.z,
                    a * b.w);
}

occaFunction inline char4  operator *  (const char4 &a, const char &b) {
  return OCCA_CHAR4(a.x * b,
                    a.y * b,
                    a.z * b,
                    a.w * b);
}

occaFunction inline char4& operator *= (      char4 &a, const char4 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  return a;
}

occaFunction inline char4& operator *= (      char4 &a, const char &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}
occaFunction inline char4  operator /  (const char4 &a, const char4 &b) {
  return OCCA_CHAR4(a.x / b.x,
                    a.y / b.y,
                    a.z / b.z,
                    a.w / b.w);
}

occaFunction inline char4  operator /  (const char &a, const char4 &b) {
  return OCCA_CHAR4(a / b.x,
                    a / b.y,
                    a / b.z,
                    a / b.w);
}

occaFunction inline char4  operator /  (const char4 &a, const char &b) {
  return OCCA_CHAR4(a.x / b,
                    a.y / b,
                    a.z / b,
                    a.w / b);
}

occaFunction inline char4& operator /= (      char4 &a, const char4 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  return a;
}

occaFunction inline char4& operator /= (      char4 &a, const char &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const char4& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w
      << "]\n";

  return out;
}
#endif

//======================================


//---[ char3 ]--------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_CHAR3 make_char3
#else
#  define OCCA_CHAR3 char3
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
typedef char4 char3;
#endif
//======================================


//---[ char8 ]--------------------------
#define OCCA_CHAR8 char8
class char8{
public:
  union { char s0, x; };
  union { char s1, y; };
  union { char s2, z; };
  union { char s3, w; };
  char s4;
  char s5;
  char s6;
  char s7;

  inline occaFunction char8() : 
    x(0),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction char8(const char &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction char8(const char &x_,
                            const char &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction char8(const char &x_,
                            const char &y_,
                            const char &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction char8(const char &x_,
                            const char &y_,
                            const char &z_,
                            const char &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction char8(const char &x_,
                            const char &y_,
                            const char &z_,
                            const char &w_,
                            const char &s4_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction char8(const char &x_,
                            const char &y_,
                            const char &z_,
                            const char &w_,
                            const char &s4_,
                            const char &s5_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(0),
    s7(0) {}

  inline occaFunction char8(const char &x_,
                            const char &y_,
                            const char &z_,
                            const char &w_,
                            const char &s4_,
                            const char &s5_,
                            const char &s6_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(0) {}

  inline occaFunction char8(const char &x_,
                            const char &y_,
                            const char &z_,
                            const char &w_,
                            const char &s4_,
                            const char &s5_,
                            const char &s6_,
                            const char &s7_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_) {}
};

occaFunction inline char8 operator + (const char8 &a) {
  return OCCA_CHAR8(+a.x,
                    +a.y,
                    +a.z,
                    +a.w,
                    +a.s4,
                    +a.s5,
                    +a.s6,
                    +a.s7);
}

occaFunction inline char8 operator ++ (char8 &a, int) {
  return OCCA_CHAR8(a.x++,
                    a.y++,
                    a.z++,
                    a.w++,
                    a.s4++,
                    a.s5++,
                    a.s6++,
                    a.s7++);
}

occaFunction inline char8& operator ++ (char8 &a) {
  ++a.x;
  ++a.y;
  ++a.z;
  ++a.w;
  ++a.s4;
  ++a.s5;
  ++a.s6;
  ++a.s7;
  return a;
}
occaFunction inline char8 operator - (const char8 &a) {
  return OCCA_CHAR8(-a.x,
                    -a.y,
                    -a.z,
                    -a.w,
                    -a.s4,
                    -a.s5,
                    -a.s6,
                    -a.s7);
}

occaFunction inline char8 operator -- (char8 &a, int) {
  return OCCA_CHAR8(a.x--,
                    a.y--,
                    a.z--,
                    a.w--,
                    a.s4--,
                    a.s5--,
                    a.s6--,
                    a.s7--);
}

occaFunction inline char8& operator -- (char8 &a) {
  --a.x;
  --a.y;
  --a.z;
  --a.w;
  --a.s4;
  --a.s5;
  --a.s6;
  --a.s7;
  return a;
}
occaFunction inline char8  operator +  (const char8 &a, const char8 &b) {
  return OCCA_CHAR8(a.x + b.x,
                    a.y + b.y,
                    a.z + b.z,
                    a.w + b.w,
                    a.s4 + b.s4,
                    a.s5 + b.s5,
                    a.s6 + b.s6,
                    a.s7 + b.s7);
}

occaFunction inline char8  operator +  (const char &a, const char8 &b) {
  return OCCA_CHAR8(a + b.x,
                    a + b.y,
                    a + b.z,
                    a + b.w,
                    a + b.s4,
                    a + b.s5,
                    a + b.s6,
                    a + b.s7);
}

occaFunction inline char8  operator +  (const char8 &a, const char &b) {
  return OCCA_CHAR8(a.x + b,
                    a.y + b,
                    a.z + b,
                    a.w + b,
                    a.s4 + b,
                    a.s5 + b,
                    a.s6 + b,
                    a.s7 + b);
}

occaFunction inline char8& operator += (      char8 &a, const char8 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  a.s4 += b.s4;
  a.s5 += b.s5;
  a.s6 += b.s6;
  a.s7 += b.s7;
  return a;
}

occaFunction inline char8& operator += (      char8 &a, const char &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  a.s4 += b;
  a.s5 += b;
  a.s6 += b;
  a.s7 += b;
  return a;
}
occaFunction inline char8  operator -  (const char8 &a, const char8 &b) {
  return OCCA_CHAR8(a.x - b.x,
                    a.y - b.y,
                    a.z - b.z,
                    a.w - b.w,
                    a.s4 - b.s4,
                    a.s5 - b.s5,
                    a.s6 - b.s6,
                    a.s7 - b.s7);
}

occaFunction inline char8  operator -  (const char &a, const char8 &b) {
  return OCCA_CHAR8(a - b.x,
                    a - b.y,
                    a - b.z,
                    a - b.w,
                    a - b.s4,
                    a - b.s5,
                    a - b.s6,
                    a - b.s7);
}

occaFunction inline char8  operator -  (const char8 &a, const char &b) {
  return OCCA_CHAR8(a.x - b,
                    a.y - b,
                    a.z - b,
                    a.w - b,
                    a.s4 - b,
                    a.s5 - b,
                    a.s6 - b,
                    a.s7 - b);
}

occaFunction inline char8& operator -= (      char8 &a, const char8 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  a.s4 -= b.s4;
  a.s5 -= b.s5;
  a.s6 -= b.s6;
  a.s7 -= b.s7;
  return a;
}

occaFunction inline char8& operator -= (      char8 &a, const char &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  a.s4 -= b;
  a.s5 -= b;
  a.s6 -= b;
  a.s7 -= b;
  return a;
}
occaFunction inline char8  operator *  (const char8 &a, const char8 &b) {
  return OCCA_CHAR8(a.x * b.x,
                    a.y * b.y,
                    a.z * b.z,
                    a.w * b.w,
                    a.s4 * b.s4,
                    a.s5 * b.s5,
                    a.s6 * b.s6,
                    a.s7 * b.s7);
}

occaFunction inline char8  operator *  (const char &a, const char8 &b) {
  return OCCA_CHAR8(a * b.x,
                    a * b.y,
                    a * b.z,
                    a * b.w,
                    a * b.s4,
                    a * b.s5,
                    a * b.s6,
                    a * b.s7);
}

occaFunction inline char8  operator *  (const char8 &a, const char &b) {
  return OCCA_CHAR8(a.x * b,
                    a.y * b,
                    a.z * b,
                    a.w * b,
                    a.s4 * b,
                    a.s5 * b,
                    a.s6 * b,
                    a.s7 * b);
}

occaFunction inline char8& operator *= (      char8 &a, const char8 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  a.s4 *= b.s4;
  a.s5 *= b.s5;
  a.s6 *= b.s6;
  a.s7 *= b.s7;
  return a;
}

occaFunction inline char8& operator *= (      char8 &a, const char &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  a.s4 *= b;
  a.s5 *= b;
  a.s6 *= b;
  a.s7 *= b;
  return a;
}
occaFunction inline char8  operator /  (const char8 &a, const char8 &b) {
  return OCCA_CHAR8(a.x / b.x,
                    a.y / b.y,
                    a.z / b.z,
                    a.w / b.w,
                    a.s4 / b.s4,
                    a.s5 / b.s5,
                    a.s6 / b.s6,
                    a.s7 / b.s7);
}

occaFunction inline char8  operator /  (const char &a, const char8 &b) {
  return OCCA_CHAR8(a / b.x,
                    a / b.y,
                    a / b.z,
                    a / b.w,
                    a / b.s4,
                    a / b.s5,
                    a / b.s6,
                    a / b.s7);
}

occaFunction inline char8  operator /  (const char8 &a, const char &b) {
  return OCCA_CHAR8(a.x / b,
                    a.y / b,
                    a.z / b,
                    a.w / b,
                    a.s4 / b,
                    a.s5 / b,
                    a.s6 / b,
                    a.s7 / b);
}

occaFunction inline char8& operator /= (      char8 &a, const char8 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  a.s4 /= b.s4;
  a.s5 /= b.s5;
  a.s6 /= b.s6;
  a.s7 /= b.s7;
  return a;
}

occaFunction inline char8& operator /= (      char8 &a, const char &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  a.s4 /= b;
  a.s5 /= b;
  a.s6 /= b;
  a.s7 /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const char8& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w << ", "
             << a.s4 << ", "
             << a.s5 << ", "
             << a.s6 << ", "
             << a.s7
      << "]\n";

  return out;
}
#endif

//======================================


//---[ char16 ]-------------------------
#define OCCA_CHAR16 char16
class char16{
public:
  union { char s0, x; };
  union { char s1, y; };
  union { char s2, z; };
  union { char s3, w; };
  char s4;
  char s5;
  char s6;
  char s7;
  char s8;
  char s9;
  char s10;
  char s11;
  char s12;
  char s13;
  char s14;
  char s15;

  inline occaFunction char16() : 
    x(0),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction char16(const char &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction char16(const char &x_,
                             const char &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction char16(const char &x_,
                             const char &y_,
                             const char &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction char16(const char &x_,
                             const char &y_,
                             const char &z_,
                             const char &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction char16(const char &x_,
                             const char &y_,
                             const char &z_,
                             const char &w_,
                             const char &s4_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction char16(const char &x_,
                             const char &y_,
                             const char &z_,
                             const char &w_,
                             const char &s4_,
                             const char &s5_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction char16(const char &x_,
                             const char &y_,
                             const char &z_,
                             const char &w_,
                             const char &s4_,
                             const char &s5_,
                             const char &s6_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction char16(const char &x_,
                             const char &y_,
                             const char &z_,
                             const char &w_,
                             const char &s4_,
                             const char &s5_,
                             const char &s6_,
                             const char &s7_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction char16(const char &x_,
                             const char &y_,
                             const char &z_,
                             const char &w_,
                             const char &s4_,
                             const char &s5_,
                             const char &s6_,
                             const char &s7_,
                             const char &s8_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction char16(const char &x_,
                             const char &y_,
                             const char &z_,
                             const char &w_,
                             const char &s4_,
                             const char &s5_,
                             const char &s6_,
                             const char &s7_,
                             const char &s8_,
                             const char &s9_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction char16(const char &x_,
                             const char &y_,
                             const char &z_,
                             const char &w_,
                             const char &s4_,
                             const char &s5_,
                             const char &s6_,
                             const char &s7_,
                             const char &s8_,
                             const char &s9_,
                             const char &s10_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction char16(const char &x_,
                             const char &y_,
                             const char &z_,
                             const char &w_,
                             const char &s4_,
                             const char &s5_,
                             const char &s6_,
                             const char &s7_,
                             const char &s8_,
                             const char &s9_,
                             const char &s10_,
                             const char &s11_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction char16(const char &x_,
                             const char &y_,
                             const char &z_,
                             const char &w_,
                             const char &s4_,
                             const char &s5_,
                             const char &s6_,
                             const char &s7_,
                             const char &s8_,
                             const char &s9_,
                             const char &s10_,
                             const char &s11_,
                             const char &s12_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction char16(const char &x_,
                             const char &y_,
                             const char &z_,
                             const char &w_,
                             const char &s4_,
                             const char &s5_,
                             const char &s6_,
                             const char &s7_,
                             const char &s8_,
                             const char &s9_,
                             const char &s10_,
                             const char &s11_,
                             const char &s12_,
                             const char &s13_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(0),
    s15(0) {}

  inline occaFunction char16(const char &x_,
                             const char &y_,
                             const char &z_,
                             const char &w_,
                             const char &s4_,
                             const char &s5_,
                             const char &s6_,
                             const char &s7_,
                             const char &s8_,
                             const char &s9_,
                             const char &s10_,
                             const char &s11_,
                             const char &s12_,
                             const char &s13_,
                             const char &s14_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(s14_),
    s15(0) {}

  inline occaFunction char16(const char &x_,
                             const char &y_,
                             const char &z_,
                             const char &w_,
                             const char &s4_,
                             const char &s5_,
                             const char &s6_,
                             const char &s7_,
                             const char &s8_,
                             const char &s9_,
                             const char &s10_,
                             const char &s11_,
                             const char &s12_,
                             const char &s13_,
                             const char &s14_,
                             const char &s15_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(s14_),
    s15(s15_) {}
};

occaFunction inline char16 operator + (const char16 &a) {
  return OCCA_CHAR16(+a.x,
                     +a.y,
                     +a.z,
                     +a.w,
                     +a.s4,
                     +a.s5,
                     +a.s6,
                     +a.s7,
                     +a.s8,
                     +a.s9,
                     +a.s10,
                     +a.s11,
                     +a.s12,
                     +a.s13,
                     +a.s14,
                     +a.s15);
}

occaFunction inline char16 operator ++ (char16 &a, int) {
  return OCCA_CHAR16(a.x++,
                     a.y++,
                     a.z++,
                     a.w++,
                     a.s4++,
                     a.s5++,
                     a.s6++,
                     a.s7++,
                     a.s8++,
                     a.s9++,
                     a.s10++,
                     a.s11++,
                     a.s12++,
                     a.s13++,
                     a.s14++,
                     a.s15++);
}

occaFunction inline char16& operator ++ (char16 &a) {
  ++a.x;
  ++a.y;
  ++a.z;
  ++a.w;
  ++a.s4;
  ++a.s5;
  ++a.s6;
  ++a.s7;
  ++a.s8;
  ++a.s9;
  ++a.s10;
  ++a.s11;
  ++a.s12;
  ++a.s13;
  ++a.s14;
  ++a.s15;
  return a;
}
occaFunction inline char16 operator - (const char16 &a) {
  return OCCA_CHAR16(-a.x,
                     -a.y,
                     -a.z,
                     -a.w,
                     -a.s4,
                     -a.s5,
                     -a.s6,
                     -a.s7,
                     -a.s8,
                     -a.s9,
                     -a.s10,
                     -a.s11,
                     -a.s12,
                     -a.s13,
                     -a.s14,
                     -a.s15);
}

occaFunction inline char16 operator -- (char16 &a, int) {
  return OCCA_CHAR16(a.x--,
                     a.y--,
                     a.z--,
                     a.w--,
                     a.s4--,
                     a.s5--,
                     a.s6--,
                     a.s7--,
                     a.s8--,
                     a.s9--,
                     a.s10--,
                     a.s11--,
                     a.s12--,
                     a.s13--,
                     a.s14--,
                     a.s15--);
}

occaFunction inline char16& operator -- (char16 &a) {
  --a.x;
  --a.y;
  --a.z;
  --a.w;
  --a.s4;
  --a.s5;
  --a.s6;
  --a.s7;
  --a.s8;
  --a.s9;
  --a.s10;
  --a.s11;
  --a.s12;
  --a.s13;
  --a.s14;
  --a.s15;
  return a;
}
occaFunction inline char16  operator +  (const char16 &a, const char16 &b) {
  return OCCA_CHAR16(a.x + b.x,
                     a.y + b.y,
                     a.z + b.z,
                     a.w + b.w,
                     a.s4 + b.s4,
                     a.s5 + b.s5,
                     a.s6 + b.s6,
                     a.s7 + b.s7,
                     a.s8 + b.s8,
                     a.s9 + b.s9,
                     a.s10 + b.s10,
                     a.s11 + b.s11,
                     a.s12 + b.s12,
                     a.s13 + b.s13,
                     a.s14 + b.s14,
                     a.s15 + b.s15);
}

occaFunction inline char16  operator +  (const char &a, const char16 &b) {
  return OCCA_CHAR16(a + b.x,
                     a + b.y,
                     a + b.z,
                     a + b.w,
                     a + b.s4,
                     a + b.s5,
                     a + b.s6,
                     a + b.s7,
                     a + b.s8,
                     a + b.s9,
                     a + b.s10,
                     a + b.s11,
                     a + b.s12,
                     a + b.s13,
                     a + b.s14,
                     a + b.s15);
}

occaFunction inline char16  operator +  (const char16 &a, const char &b) {
  return OCCA_CHAR16(a.x + b,
                     a.y + b,
                     a.z + b,
                     a.w + b,
                     a.s4 + b,
                     a.s5 + b,
                     a.s6 + b,
                     a.s7 + b,
                     a.s8 + b,
                     a.s9 + b,
                     a.s10 + b,
                     a.s11 + b,
                     a.s12 + b,
                     a.s13 + b,
                     a.s14 + b,
                     a.s15 + b);
}

occaFunction inline char16& operator += (      char16 &a, const char16 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  a.s4 += b.s4;
  a.s5 += b.s5;
  a.s6 += b.s6;
  a.s7 += b.s7;
  a.s8 += b.s8;
  a.s9 += b.s9;
  a.s10 += b.s10;
  a.s11 += b.s11;
  a.s12 += b.s12;
  a.s13 += b.s13;
  a.s14 += b.s14;
  a.s15 += b.s15;
  return a;
}

occaFunction inline char16& operator += (      char16 &a, const char &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  a.s4 += b;
  a.s5 += b;
  a.s6 += b;
  a.s7 += b;
  a.s8 += b;
  a.s9 += b;
  a.s10 += b;
  a.s11 += b;
  a.s12 += b;
  a.s13 += b;
  a.s14 += b;
  a.s15 += b;
  return a;
}
occaFunction inline char16  operator -  (const char16 &a, const char16 &b) {
  return OCCA_CHAR16(a.x - b.x,
                     a.y - b.y,
                     a.z - b.z,
                     a.w - b.w,
                     a.s4 - b.s4,
                     a.s5 - b.s5,
                     a.s6 - b.s6,
                     a.s7 - b.s7,
                     a.s8 - b.s8,
                     a.s9 - b.s9,
                     a.s10 - b.s10,
                     a.s11 - b.s11,
                     a.s12 - b.s12,
                     a.s13 - b.s13,
                     a.s14 - b.s14,
                     a.s15 - b.s15);
}

occaFunction inline char16  operator -  (const char &a, const char16 &b) {
  return OCCA_CHAR16(a - b.x,
                     a - b.y,
                     a - b.z,
                     a - b.w,
                     a - b.s4,
                     a - b.s5,
                     a - b.s6,
                     a - b.s7,
                     a - b.s8,
                     a - b.s9,
                     a - b.s10,
                     a - b.s11,
                     a - b.s12,
                     a - b.s13,
                     a - b.s14,
                     a - b.s15);
}

occaFunction inline char16  operator -  (const char16 &a, const char &b) {
  return OCCA_CHAR16(a.x - b,
                     a.y - b,
                     a.z - b,
                     a.w - b,
                     a.s4 - b,
                     a.s5 - b,
                     a.s6 - b,
                     a.s7 - b,
                     a.s8 - b,
                     a.s9 - b,
                     a.s10 - b,
                     a.s11 - b,
                     a.s12 - b,
                     a.s13 - b,
                     a.s14 - b,
                     a.s15 - b);
}

occaFunction inline char16& operator -= (      char16 &a, const char16 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  a.s4 -= b.s4;
  a.s5 -= b.s5;
  a.s6 -= b.s6;
  a.s7 -= b.s7;
  a.s8 -= b.s8;
  a.s9 -= b.s9;
  a.s10 -= b.s10;
  a.s11 -= b.s11;
  a.s12 -= b.s12;
  a.s13 -= b.s13;
  a.s14 -= b.s14;
  a.s15 -= b.s15;
  return a;
}

occaFunction inline char16& operator -= (      char16 &a, const char &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  a.s4 -= b;
  a.s5 -= b;
  a.s6 -= b;
  a.s7 -= b;
  a.s8 -= b;
  a.s9 -= b;
  a.s10 -= b;
  a.s11 -= b;
  a.s12 -= b;
  a.s13 -= b;
  a.s14 -= b;
  a.s15 -= b;
  return a;
}
occaFunction inline char16  operator *  (const char16 &a, const char16 &b) {
  return OCCA_CHAR16(a.x * b.x,
                     a.y * b.y,
                     a.z * b.z,
                     a.w * b.w,
                     a.s4 * b.s4,
                     a.s5 * b.s5,
                     a.s6 * b.s6,
                     a.s7 * b.s7,
                     a.s8 * b.s8,
                     a.s9 * b.s9,
                     a.s10 * b.s10,
                     a.s11 * b.s11,
                     a.s12 * b.s12,
                     a.s13 * b.s13,
                     a.s14 * b.s14,
                     a.s15 * b.s15);
}

occaFunction inline char16  operator *  (const char &a, const char16 &b) {
  return OCCA_CHAR16(a * b.x,
                     a * b.y,
                     a * b.z,
                     a * b.w,
                     a * b.s4,
                     a * b.s5,
                     a * b.s6,
                     a * b.s7,
                     a * b.s8,
                     a * b.s9,
                     a * b.s10,
                     a * b.s11,
                     a * b.s12,
                     a * b.s13,
                     a * b.s14,
                     a * b.s15);
}

occaFunction inline char16  operator *  (const char16 &a, const char &b) {
  return OCCA_CHAR16(a.x * b,
                     a.y * b,
                     a.z * b,
                     a.w * b,
                     a.s4 * b,
                     a.s5 * b,
                     a.s6 * b,
                     a.s7 * b,
                     a.s8 * b,
                     a.s9 * b,
                     a.s10 * b,
                     a.s11 * b,
                     a.s12 * b,
                     a.s13 * b,
                     a.s14 * b,
                     a.s15 * b);
}

occaFunction inline char16& operator *= (      char16 &a, const char16 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  a.s4 *= b.s4;
  a.s5 *= b.s5;
  a.s6 *= b.s6;
  a.s7 *= b.s7;
  a.s8 *= b.s8;
  a.s9 *= b.s9;
  a.s10 *= b.s10;
  a.s11 *= b.s11;
  a.s12 *= b.s12;
  a.s13 *= b.s13;
  a.s14 *= b.s14;
  a.s15 *= b.s15;
  return a;
}

occaFunction inline char16& operator *= (      char16 &a, const char &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  a.s4 *= b;
  a.s5 *= b;
  a.s6 *= b;
  a.s7 *= b;
  a.s8 *= b;
  a.s9 *= b;
  a.s10 *= b;
  a.s11 *= b;
  a.s12 *= b;
  a.s13 *= b;
  a.s14 *= b;
  a.s15 *= b;
  return a;
}
occaFunction inline char16  operator /  (const char16 &a, const char16 &b) {
  return OCCA_CHAR16(a.x / b.x,
                     a.y / b.y,
                     a.z / b.z,
                     a.w / b.w,
                     a.s4 / b.s4,
                     a.s5 / b.s5,
                     a.s6 / b.s6,
                     a.s7 / b.s7,
                     a.s8 / b.s8,
                     a.s9 / b.s9,
                     a.s10 / b.s10,
                     a.s11 / b.s11,
                     a.s12 / b.s12,
                     a.s13 / b.s13,
                     a.s14 / b.s14,
                     a.s15 / b.s15);
}

occaFunction inline char16  operator /  (const char &a, const char16 &b) {
  return OCCA_CHAR16(a / b.x,
                     a / b.y,
                     a / b.z,
                     a / b.w,
                     a / b.s4,
                     a / b.s5,
                     a / b.s6,
                     a / b.s7,
                     a / b.s8,
                     a / b.s9,
                     a / b.s10,
                     a / b.s11,
                     a / b.s12,
                     a / b.s13,
                     a / b.s14,
                     a / b.s15);
}

occaFunction inline char16  operator /  (const char16 &a, const char &b) {
  return OCCA_CHAR16(a.x / b,
                     a.y / b,
                     a.z / b,
                     a.w / b,
                     a.s4 / b,
                     a.s5 / b,
                     a.s6 / b,
                     a.s7 / b,
                     a.s8 / b,
                     a.s9 / b,
                     a.s10 / b,
                     a.s11 / b,
                     a.s12 / b,
                     a.s13 / b,
                     a.s14 / b,
                     a.s15 / b);
}

occaFunction inline char16& operator /= (      char16 &a, const char16 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  a.s4 /= b.s4;
  a.s5 /= b.s5;
  a.s6 /= b.s6;
  a.s7 /= b.s7;
  a.s8 /= b.s8;
  a.s9 /= b.s9;
  a.s10 /= b.s10;
  a.s11 /= b.s11;
  a.s12 /= b.s12;
  a.s13 /= b.s13;
  a.s14 /= b.s14;
  a.s15 /= b.s15;
  return a;
}

occaFunction inline char16& operator /= (      char16 &a, const char &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  a.s4 /= b;
  a.s5 /= b;
  a.s6 /= b;
  a.s7 /= b;
  a.s8 /= b;
  a.s9 /= b;
  a.s10 /= b;
  a.s11 /= b;
  a.s12 /= b;
  a.s13 /= b;
  a.s14 /= b;
  a.s15 /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const char16& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w << ", "
             << a.s4 << ", "
             << a.s5 << ", "
             << a.s6 << ", "
             << a.s7 << ", "
             << a.s8 << ", "
             << a.s9 << ", "
             << a.s10 << ", "
             << a.s11 << ", "
             << a.s12 << ", "
             << a.s13 << ", "
             << a.s14 << ", "
             << a.s15
      << "]\n";

  return out;
}
#endif

//======================================


//---[ short2 ]-------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_SHORT2 make_short2
#else
#  define OCCA_SHORT2 short2
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
class short2{
public:
  union { short s0, x; };
  union { short s1, y; };

  inline occaFunction short2() : 
    x(0),
    y(0) {}

  inline occaFunction short2(const short &x_) : 
    x(x_),
    y(0) {}

  inline occaFunction short2(const short &x_,
                             const short &y_) : 
    x(x_),
    y(y_) {}
};
#endif

occaFunction inline short2 operator + (const short2 &a) {
  return OCCA_SHORT2(+a.x,
                     +a.y);
}

occaFunction inline short2 operator ++ (short2 &a, int) {
  return OCCA_SHORT2(a.x++,
                     a.y++);
}

occaFunction inline short2& operator ++ (short2 &a) {
  ++a.x;
  ++a.y;
  return a;
}
occaFunction inline short2 operator - (const short2 &a) {
  return OCCA_SHORT2(-a.x,
                     -a.y);
}

occaFunction inline short2 operator -- (short2 &a, int) {
  return OCCA_SHORT2(a.x--,
                     a.y--);
}

occaFunction inline short2& operator -- (short2 &a) {
  --a.x;
  --a.y;
  return a;
}
occaFunction inline short2  operator +  (const short2 &a, const short2 &b) {
  return OCCA_SHORT2(a.x + b.x,
                     a.y + b.y);
}

occaFunction inline short2  operator +  (const short &a, const short2 &b) {
  return OCCA_SHORT2(a + b.x,
                     a + b.y);
}

occaFunction inline short2  operator +  (const short2 &a, const short &b) {
  return OCCA_SHORT2(a.x + b,
                     a.y + b);
}

occaFunction inline short2& operator += (      short2 &a, const short2 &b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}

occaFunction inline short2& operator += (      short2 &a, const short &b) {
  a.x += b;
  a.y += b;
  return a;
}
occaFunction inline short2  operator -  (const short2 &a, const short2 &b) {
  return OCCA_SHORT2(a.x - b.x,
                     a.y - b.y);
}

occaFunction inline short2  operator -  (const short &a, const short2 &b) {
  return OCCA_SHORT2(a - b.x,
                     a - b.y);
}

occaFunction inline short2  operator -  (const short2 &a, const short &b) {
  return OCCA_SHORT2(a.x - b,
                     a.y - b);
}

occaFunction inline short2& operator -= (      short2 &a, const short2 &b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

occaFunction inline short2& operator -= (      short2 &a, const short &b) {
  a.x -= b;
  a.y -= b;
  return a;
}
occaFunction inline short2  operator *  (const short2 &a, const short2 &b) {
  return OCCA_SHORT2(a.x * b.x,
                     a.y * b.y);
}

occaFunction inline short2  operator *  (const short &a, const short2 &b) {
  return OCCA_SHORT2(a * b.x,
                     a * b.y);
}

occaFunction inline short2  operator *  (const short2 &a, const short &b) {
  return OCCA_SHORT2(a.x * b,
                     a.y * b);
}

occaFunction inline short2& operator *= (      short2 &a, const short2 &b) {
  a.x *= b.x;
  a.y *= b.y;
  return a;
}

occaFunction inline short2& operator *= (      short2 &a, const short &b) {
  a.x *= b;
  a.y *= b;
  return a;
}
occaFunction inline short2  operator /  (const short2 &a, const short2 &b) {
  return OCCA_SHORT2(a.x / b.x,
                     a.y / b.y);
}

occaFunction inline short2  operator /  (const short &a, const short2 &b) {
  return OCCA_SHORT2(a / b.x,
                     a / b.y);
}

occaFunction inline short2  operator /  (const short2 &a, const short &b) {
  return OCCA_SHORT2(a.x / b,
                     a.y / b);
}

occaFunction inline short2& operator /= (      short2 &a, const short2 &b) {
  a.x /= b.x;
  a.y /= b.y;
  return a;
}

occaFunction inline short2& operator /= (      short2 &a, const short &b) {
  a.x /= b;
  a.y /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const short2& a) {
  out << "[" << a.x << ", "
             << a.y
      << "]\n";

  return out;
}
#endif

//======================================


//---[ short4 ]-------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_SHORT4 make_short4
#else
#  define OCCA_SHORT4 short4
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
class short4{
public:
  union { short s0, x; };
  union { short s1, y; };
  union { short s2, z; };
  union { short s3, w; };

  inline occaFunction short4() : 
    x(0),
    y(0),
    z(0),
    w(0) {}

  inline occaFunction short4(const short &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0) {}

  inline occaFunction short4(const short &x_,
                             const short &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0) {}

  inline occaFunction short4(const short &x_,
                             const short &y_,
                             const short &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0) {}

  inline occaFunction short4(const short &x_,
                             const short &y_,
                             const short &z_,
                             const short &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_) {}
};
#endif

occaFunction inline short4 operator + (const short4 &a) {
  return OCCA_SHORT4(+a.x,
                     +a.y,
                     +a.z,
                     +a.w);
}

occaFunction inline short4 operator ++ (short4 &a, int) {
  return OCCA_SHORT4(a.x++,
                     a.y++,
                     a.z++,
                     a.w++);
}

occaFunction inline short4& operator ++ (short4 &a) {
  ++a.x;
  ++a.y;
  ++a.z;
  ++a.w;
  return a;
}
occaFunction inline short4 operator - (const short4 &a) {
  return OCCA_SHORT4(-a.x,
                     -a.y,
                     -a.z,
                     -a.w);
}

occaFunction inline short4 operator -- (short4 &a, int) {
  return OCCA_SHORT4(a.x--,
                     a.y--,
                     a.z--,
                     a.w--);
}

occaFunction inline short4& operator -- (short4 &a) {
  --a.x;
  --a.y;
  --a.z;
  --a.w;
  return a;
}
occaFunction inline short4  operator +  (const short4 &a, const short4 &b) {
  return OCCA_SHORT4(a.x + b.x,
                     a.y + b.y,
                     a.z + b.z,
                     a.w + b.w);
}

occaFunction inline short4  operator +  (const short &a, const short4 &b) {
  return OCCA_SHORT4(a + b.x,
                     a + b.y,
                     a + b.z,
                     a + b.w);
}

occaFunction inline short4  operator +  (const short4 &a, const short &b) {
  return OCCA_SHORT4(a.x + b,
                     a.y + b,
                     a.z + b,
                     a.w + b);
}

occaFunction inline short4& operator += (      short4 &a, const short4 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

occaFunction inline short4& operator += (      short4 &a, const short &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  return a;
}
occaFunction inline short4  operator -  (const short4 &a, const short4 &b) {
  return OCCA_SHORT4(a.x - b.x,
                     a.y - b.y,
                     a.z - b.z,
                     a.w - b.w);
}

occaFunction inline short4  operator -  (const short &a, const short4 &b) {
  return OCCA_SHORT4(a - b.x,
                     a - b.y,
                     a - b.z,
                     a - b.w);
}

occaFunction inline short4  operator -  (const short4 &a, const short &b) {
  return OCCA_SHORT4(a.x - b,
                     a.y - b,
                     a.z - b,
                     a.w - b);
}

occaFunction inline short4& operator -= (      short4 &a, const short4 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  return a;
}

occaFunction inline short4& operator -= (      short4 &a, const short &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  return a;
}
occaFunction inline short4  operator *  (const short4 &a, const short4 &b) {
  return OCCA_SHORT4(a.x * b.x,
                     a.y * b.y,
                     a.z * b.z,
                     a.w * b.w);
}

occaFunction inline short4  operator *  (const short &a, const short4 &b) {
  return OCCA_SHORT4(a * b.x,
                     a * b.y,
                     a * b.z,
                     a * b.w);
}

occaFunction inline short4  operator *  (const short4 &a, const short &b) {
  return OCCA_SHORT4(a.x * b,
                     a.y * b,
                     a.z * b,
                     a.w * b);
}

occaFunction inline short4& operator *= (      short4 &a, const short4 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  return a;
}

occaFunction inline short4& operator *= (      short4 &a, const short &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}
occaFunction inline short4  operator /  (const short4 &a, const short4 &b) {
  return OCCA_SHORT4(a.x / b.x,
                     a.y / b.y,
                     a.z / b.z,
                     a.w / b.w);
}

occaFunction inline short4  operator /  (const short &a, const short4 &b) {
  return OCCA_SHORT4(a / b.x,
                     a / b.y,
                     a / b.z,
                     a / b.w);
}

occaFunction inline short4  operator /  (const short4 &a, const short &b) {
  return OCCA_SHORT4(a.x / b,
                     a.y / b,
                     a.z / b,
                     a.w / b);
}

occaFunction inline short4& operator /= (      short4 &a, const short4 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  return a;
}

occaFunction inline short4& operator /= (      short4 &a, const short &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const short4& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w
      << "]\n";

  return out;
}
#endif

//======================================


//---[ short3 ]-------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_SHORT3 make_short3
#else
#  define OCCA_SHORT3 short3
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
typedef short4 short3;
#endif
//======================================


//---[ short8 ]-------------------------
#define OCCA_SHORT8 short8
class short8{
public:
  union { short s0, x; };
  union { short s1, y; };
  union { short s2, z; };
  union { short s3, w; };
  short s4;
  short s5;
  short s6;
  short s7;

  inline occaFunction short8() : 
    x(0),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction short8(const short &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction short8(const short &x_,
                             const short &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction short8(const short &x_,
                             const short &y_,
                             const short &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction short8(const short &x_,
                             const short &y_,
                             const short &z_,
                             const short &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction short8(const short &x_,
                             const short &y_,
                             const short &z_,
                             const short &w_,
                             const short &s4_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction short8(const short &x_,
                             const short &y_,
                             const short &z_,
                             const short &w_,
                             const short &s4_,
                             const short &s5_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(0),
    s7(0) {}

  inline occaFunction short8(const short &x_,
                             const short &y_,
                             const short &z_,
                             const short &w_,
                             const short &s4_,
                             const short &s5_,
                             const short &s6_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(0) {}

  inline occaFunction short8(const short &x_,
                             const short &y_,
                             const short &z_,
                             const short &w_,
                             const short &s4_,
                             const short &s5_,
                             const short &s6_,
                             const short &s7_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_) {}
};

occaFunction inline short8 operator + (const short8 &a) {
  return OCCA_SHORT8(+a.x,
                     +a.y,
                     +a.z,
                     +a.w,
                     +a.s4,
                     +a.s5,
                     +a.s6,
                     +a.s7);
}

occaFunction inline short8 operator ++ (short8 &a, int) {
  return OCCA_SHORT8(a.x++,
                     a.y++,
                     a.z++,
                     a.w++,
                     a.s4++,
                     a.s5++,
                     a.s6++,
                     a.s7++);
}

occaFunction inline short8& operator ++ (short8 &a) {
  ++a.x;
  ++a.y;
  ++a.z;
  ++a.w;
  ++a.s4;
  ++a.s5;
  ++a.s6;
  ++a.s7;
  return a;
}
occaFunction inline short8 operator - (const short8 &a) {
  return OCCA_SHORT8(-a.x,
                     -a.y,
                     -a.z,
                     -a.w,
                     -a.s4,
                     -a.s5,
                     -a.s6,
                     -a.s7);
}

occaFunction inline short8 operator -- (short8 &a, int) {
  return OCCA_SHORT8(a.x--,
                     a.y--,
                     a.z--,
                     a.w--,
                     a.s4--,
                     a.s5--,
                     a.s6--,
                     a.s7--);
}

occaFunction inline short8& operator -- (short8 &a) {
  --a.x;
  --a.y;
  --a.z;
  --a.w;
  --a.s4;
  --a.s5;
  --a.s6;
  --a.s7;
  return a;
}
occaFunction inline short8  operator +  (const short8 &a, const short8 &b) {
  return OCCA_SHORT8(a.x + b.x,
                     a.y + b.y,
                     a.z + b.z,
                     a.w + b.w,
                     a.s4 + b.s4,
                     a.s5 + b.s5,
                     a.s6 + b.s6,
                     a.s7 + b.s7);
}

occaFunction inline short8  operator +  (const short &a, const short8 &b) {
  return OCCA_SHORT8(a + b.x,
                     a + b.y,
                     a + b.z,
                     a + b.w,
                     a + b.s4,
                     a + b.s5,
                     a + b.s6,
                     a + b.s7);
}

occaFunction inline short8  operator +  (const short8 &a, const short &b) {
  return OCCA_SHORT8(a.x + b,
                     a.y + b,
                     a.z + b,
                     a.w + b,
                     a.s4 + b,
                     a.s5 + b,
                     a.s6 + b,
                     a.s7 + b);
}

occaFunction inline short8& operator += (      short8 &a, const short8 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  a.s4 += b.s4;
  a.s5 += b.s5;
  a.s6 += b.s6;
  a.s7 += b.s7;
  return a;
}

occaFunction inline short8& operator += (      short8 &a, const short &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  a.s4 += b;
  a.s5 += b;
  a.s6 += b;
  a.s7 += b;
  return a;
}
occaFunction inline short8  operator -  (const short8 &a, const short8 &b) {
  return OCCA_SHORT8(a.x - b.x,
                     a.y - b.y,
                     a.z - b.z,
                     a.w - b.w,
                     a.s4 - b.s4,
                     a.s5 - b.s5,
                     a.s6 - b.s6,
                     a.s7 - b.s7);
}

occaFunction inline short8  operator -  (const short &a, const short8 &b) {
  return OCCA_SHORT8(a - b.x,
                     a - b.y,
                     a - b.z,
                     a - b.w,
                     a - b.s4,
                     a - b.s5,
                     a - b.s6,
                     a - b.s7);
}

occaFunction inline short8  operator -  (const short8 &a, const short &b) {
  return OCCA_SHORT8(a.x - b,
                     a.y - b,
                     a.z - b,
                     a.w - b,
                     a.s4 - b,
                     a.s5 - b,
                     a.s6 - b,
                     a.s7 - b);
}

occaFunction inline short8& operator -= (      short8 &a, const short8 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  a.s4 -= b.s4;
  a.s5 -= b.s5;
  a.s6 -= b.s6;
  a.s7 -= b.s7;
  return a;
}

occaFunction inline short8& operator -= (      short8 &a, const short &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  a.s4 -= b;
  a.s5 -= b;
  a.s6 -= b;
  a.s7 -= b;
  return a;
}
occaFunction inline short8  operator *  (const short8 &a, const short8 &b) {
  return OCCA_SHORT8(a.x * b.x,
                     a.y * b.y,
                     a.z * b.z,
                     a.w * b.w,
                     a.s4 * b.s4,
                     a.s5 * b.s5,
                     a.s6 * b.s6,
                     a.s7 * b.s7);
}

occaFunction inline short8  operator *  (const short &a, const short8 &b) {
  return OCCA_SHORT8(a * b.x,
                     a * b.y,
                     a * b.z,
                     a * b.w,
                     a * b.s4,
                     a * b.s5,
                     a * b.s6,
                     a * b.s7);
}

occaFunction inline short8  operator *  (const short8 &a, const short &b) {
  return OCCA_SHORT8(a.x * b,
                     a.y * b,
                     a.z * b,
                     a.w * b,
                     a.s4 * b,
                     a.s5 * b,
                     a.s6 * b,
                     a.s7 * b);
}

occaFunction inline short8& operator *= (      short8 &a, const short8 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  a.s4 *= b.s4;
  a.s5 *= b.s5;
  a.s6 *= b.s6;
  a.s7 *= b.s7;
  return a;
}

occaFunction inline short8& operator *= (      short8 &a, const short &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  a.s4 *= b;
  a.s5 *= b;
  a.s6 *= b;
  a.s7 *= b;
  return a;
}
occaFunction inline short8  operator /  (const short8 &a, const short8 &b) {
  return OCCA_SHORT8(a.x / b.x,
                     a.y / b.y,
                     a.z / b.z,
                     a.w / b.w,
                     a.s4 / b.s4,
                     a.s5 / b.s5,
                     a.s6 / b.s6,
                     a.s7 / b.s7);
}

occaFunction inline short8  operator /  (const short &a, const short8 &b) {
  return OCCA_SHORT8(a / b.x,
                     a / b.y,
                     a / b.z,
                     a / b.w,
                     a / b.s4,
                     a / b.s5,
                     a / b.s6,
                     a / b.s7);
}

occaFunction inline short8  operator /  (const short8 &a, const short &b) {
  return OCCA_SHORT8(a.x / b,
                     a.y / b,
                     a.z / b,
                     a.w / b,
                     a.s4 / b,
                     a.s5 / b,
                     a.s6 / b,
                     a.s7 / b);
}

occaFunction inline short8& operator /= (      short8 &a, const short8 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  a.s4 /= b.s4;
  a.s5 /= b.s5;
  a.s6 /= b.s6;
  a.s7 /= b.s7;
  return a;
}

occaFunction inline short8& operator /= (      short8 &a, const short &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  a.s4 /= b;
  a.s5 /= b;
  a.s6 /= b;
  a.s7 /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const short8& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w << ", "
             << a.s4 << ", "
             << a.s5 << ", "
             << a.s6 << ", "
             << a.s7
      << "]\n";

  return out;
}
#endif

//======================================


//---[ short16 ]------------------------
#define OCCA_SHORT16 short16
class short16{
public:
  union { short s0, x; };
  union { short s1, y; };
  union { short s2, z; };
  union { short s3, w; };
  short s4;
  short s5;
  short s6;
  short s7;
  short s8;
  short s9;
  short s10;
  short s11;
  short s12;
  short s13;
  short s14;
  short s15;

  inline occaFunction short16() : 
    x(0),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction short16(const short &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction short16(const short &x_,
                              const short &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction short16(const short &x_,
                              const short &y_,
                              const short &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction short16(const short &x_,
                              const short &y_,
                              const short &z_,
                              const short &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction short16(const short &x_,
                              const short &y_,
                              const short &z_,
                              const short &w_,
                              const short &s4_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction short16(const short &x_,
                              const short &y_,
                              const short &z_,
                              const short &w_,
                              const short &s4_,
                              const short &s5_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction short16(const short &x_,
                              const short &y_,
                              const short &z_,
                              const short &w_,
                              const short &s4_,
                              const short &s5_,
                              const short &s6_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction short16(const short &x_,
                              const short &y_,
                              const short &z_,
                              const short &w_,
                              const short &s4_,
                              const short &s5_,
                              const short &s6_,
                              const short &s7_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction short16(const short &x_,
                              const short &y_,
                              const short &z_,
                              const short &w_,
                              const short &s4_,
                              const short &s5_,
                              const short &s6_,
                              const short &s7_,
                              const short &s8_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction short16(const short &x_,
                              const short &y_,
                              const short &z_,
                              const short &w_,
                              const short &s4_,
                              const short &s5_,
                              const short &s6_,
                              const short &s7_,
                              const short &s8_,
                              const short &s9_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction short16(const short &x_,
                              const short &y_,
                              const short &z_,
                              const short &w_,
                              const short &s4_,
                              const short &s5_,
                              const short &s6_,
                              const short &s7_,
                              const short &s8_,
                              const short &s9_,
                              const short &s10_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction short16(const short &x_,
                              const short &y_,
                              const short &z_,
                              const short &w_,
                              const short &s4_,
                              const short &s5_,
                              const short &s6_,
                              const short &s7_,
                              const short &s8_,
                              const short &s9_,
                              const short &s10_,
                              const short &s11_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction short16(const short &x_,
                              const short &y_,
                              const short &z_,
                              const short &w_,
                              const short &s4_,
                              const short &s5_,
                              const short &s6_,
                              const short &s7_,
                              const short &s8_,
                              const short &s9_,
                              const short &s10_,
                              const short &s11_,
                              const short &s12_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction short16(const short &x_,
                              const short &y_,
                              const short &z_,
                              const short &w_,
                              const short &s4_,
                              const short &s5_,
                              const short &s6_,
                              const short &s7_,
                              const short &s8_,
                              const short &s9_,
                              const short &s10_,
                              const short &s11_,
                              const short &s12_,
                              const short &s13_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(0),
    s15(0) {}

  inline occaFunction short16(const short &x_,
                              const short &y_,
                              const short &z_,
                              const short &w_,
                              const short &s4_,
                              const short &s5_,
                              const short &s6_,
                              const short &s7_,
                              const short &s8_,
                              const short &s9_,
                              const short &s10_,
                              const short &s11_,
                              const short &s12_,
                              const short &s13_,
                              const short &s14_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(s14_),
    s15(0) {}

  inline occaFunction short16(const short &x_,
                              const short &y_,
                              const short &z_,
                              const short &w_,
                              const short &s4_,
                              const short &s5_,
                              const short &s6_,
                              const short &s7_,
                              const short &s8_,
                              const short &s9_,
                              const short &s10_,
                              const short &s11_,
                              const short &s12_,
                              const short &s13_,
                              const short &s14_,
                              const short &s15_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(s14_),
    s15(s15_) {}
};

occaFunction inline short16 operator + (const short16 &a) {
  return OCCA_SHORT16(+a.x,
                      +a.y,
                      +a.z,
                      +a.w,
                      +a.s4,
                      +a.s5,
                      +a.s6,
                      +a.s7,
                      +a.s8,
                      +a.s9,
                      +a.s10,
                      +a.s11,
                      +a.s12,
                      +a.s13,
                      +a.s14,
                      +a.s15);
}

occaFunction inline short16 operator ++ (short16 &a, int) {
  return OCCA_SHORT16(a.x++,
                      a.y++,
                      a.z++,
                      a.w++,
                      a.s4++,
                      a.s5++,
                      a.s6++,
                      a.s7++,
                      a.s8++,
                      a.s9++,
                      a.s10++,
                      a.s11++,
                      a.s12++,
                      a.s13++,
                      a.s14++,
                      a.s15++);
}

occaFunction inline short16& operator ++ (short16 &a) {
  ++a.x;
  ++a.y;
  ++a.z;
  ++a.w;
  ++a.s4;
  ++a.s5;
  ++a.s6;
  ++a.s7;
  ++a.s8;
  ++a.s9;
  ++a.s10;
  ++a.s11;
  ++a.s12;
  ++a.s13;
  ++a.s14;
  ++a.s15;
  return a;
}
occaFunction inline short16 operator - (const short16 &a) {
  return OCCA_SHORT16(-a.x,
                      -a.y,
                      -a.z,
                      -a.w,
                      -a.s4,
                      -a.s5,
                      -a.s6,
                      -a.s7,
                      -a.s8,
                      -a.s9,
                      -a.s10,
                      -a.s11,
                      -a.s12,
                      -a.s13,
                      -a.s14,
                      -a.s15);
}

occaFunction inline short16 operator -- (short16 &a, int) {
  return OCCA_SHORT16(a.x--,
                      a.y--,
                      a.z--,
                      a.w--,
                      a.s4--,
                      a.s5--,
                      a.s6--,
                      a.s7--,
                      a.s8--,
                      a.s9--,
                      a.s10--,
                      a.s11--,
                      a.s12--,
                      a.s13--,
                      a.s14--,
                      a.s15--);
}

occaFunction inline short16& operator -- (short16 &a) {
  --a.x;
  --a.y;
  --a.z;
  --a.w;
  --a.s4;
  --a.s5;
  --a.s6;
  --a.s7;
  --a.s8;
  --a.s9;
  --a.s10;
  --a.s11;
  --a.s12;
  --a.s13;
  --a.s14;
  --a.s15;
  return a;
}
occaFunction inline short16  operator +  (const short16 &a, const short16 &b) {
  return OCCA_SHORT16(a.x + b.x,
                      a.y + b.y,
                      a.z + b.z,
                      a.w + b.w,
                      a.s4 + b.s4,
                      a.s5 + b.s5,
                      a.s6 + b.s6,
                      a.s7 + b.s7,
                      a.s8 + b.s8,
                      a.s9 + b.s9,
                      a.s10 + b.s10,
                      a.s11 + b.s11,
                      a.s12 + b.s12,
                      a.s13 + b.s13,
                      a.s14 + b.s14,
                      a.s15 + b.s15);
}

occaFunction inline short16  operator +  (const short &a, const short16 &b) {
  return OCCA_SHORT16(a + b.x,
                      a + b.y,
                      a + b.z,
                      a + b.w,
                      a + b.s4,
                      a + b.s5,
                      a + b.s6,
                      a + b.s7,
                      a + b.s8,
                      a + b.s9,
                      a + b.s10,
                      a + b.s11,
                      a + b.s12,
                      a + b.s13,
                      a + b.s14,
                      a + b.s15);
}

occaFunction inline short16  operator +  (const short16 &a, const short &b) {
  return OCCA_SHORT16(a.x + b,
                      a.y + b,
                      a.z + b,
                      a.w + b,
                      a.s4 + b,
                      a.s5 + b,
                      a.s6 + b,
                      a.s7 + b,
                      a.s8 + b,
                      a.s9 + b,
                      a.s10 + b,
                      a.s11 + b,
                      a.s12 + b,
                      a.s13 + b,
                      a.s14 + b,
                      a.s15 + b);
}

occaFunction inline short16& operator += (      short16 &a, const short16 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  a.s4 += b.s4;
  a.s5 += b.s5;
  a.s6 += b.s6;
  a.s7 += b.s7;
  a.s8 += b.s8;
  a.s9 += b.s9;
  a.s10 += b.s10;
  a.s11 += b.s11;
  a.s12 += b.s12;
  a.s13 += b.s13;
  a.s14 += b.s14;
  a.s15 += b.s15;
  return a;
}

occaFunction inline short16& operator += (      short16 &a, const short &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  a.s4 += b;
  a.s5 += b;
  a.s6 += b;
  a.s7 += b;
  a.s8 += b;
  a.s9 += b;
  a.s10 += b;
  a.s11 += b;
  a.s12 += b;
  a.s13 += b;
  a.s14 += b;
  a.s15 += b;
  return a;
}
occaFunction inline short16  operator -  (const short16 &a, const short16 &b) {
  return OCCA_SHORT16(a.x - b.x,
                      a.y - b.y,
                      a.z - b.z,
                      a.w - b.w,
                      a.s4 - b.s4,
                      a.s5 - b.s5,
                      a.s6 - b.s6,
                      a.s7 - b.s7,
                      a.s8 - b.s8,
                      a.s9 - b.s9,
                      a.s10 - b.s10,
                      a.s11 - b.s11,
                      a.s12 - b.s12,
                      a.s13 - b.s13,
                      a.s14 - b.s14,
                      a.s15 - b.s15);
}

occaFunction inline short16  operator -  (const short &a, const short16 &b) {
  return OCCA_SHORT16(a - b.x,
                      a - b.y,
                      a - b.z,
                      a - b.w,
                      a - b.s4,
                      a - b.s5,
                      a - b.s6,
                      a - b.s7,
                      a - b.s8,
                      a - b.s9,
                      a - b.s10,
                      a - b.s11,
                      a - b.s12,
                      a - b.s13,
                      a - b.s14,
                      a - b.s15);
}

occaFunction inline short16  operator -  (const short16 &a, const short &b) {
  return OCCA_SHORT16(a.x - b,
                      a.y - b,
                      a.z - b,
                      a.w - b,
                      a.s4 - b,
                      a.s5 - b,
                      a.s6 - b,
                      a.s7 - b,
                      a.s8 - b,
                      a.s9 - b,
                      a.s10 - b,
                      a.s11 - b,
                      a.s12 - b,
                      a.s13 - b,
                      a.s14 - b,
                      a.s15 - b);
}

occaFunction inline short16& operator -= (      short16 &a, const short16 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  a.s4 -= b.s4;
  a.s5 -= b.s5;
  a.s6 -= b.s6;
  a.s7 -= b.s7;
  a.s8 -= b.s8;
  a.s9 -= b.s9;
  a.s10 -= b.s10;
  a.s11 -= b.s11;
  a.s12 -= b.s12;
  a.s13 -= b.s13;
  a.s14 -= b.s14;
  a.s15 -= b.s15;
  return a;
}

occaFunction inline short16& operator -= (      short16 &a, const short &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  a.s4 -= b;
  a.s5 -= b;
  a.s6 -= b;
  a.s7 -= b;
  a.s8 -= b;
  a.s9 -= b;
  a.s10 -= b;
  a.s11 -= b;
  a.s12 -= b;
  a.s13 -= b;
  a.s14 -= b;
  a.s15 -= b;
  return a;
}
occaFunction inline short16  operator *  (const short16 &a, const short16 &b) {
  return OCCA_SHORT16(a.x * b.x,
                      a.y * b.y,
                      a.z * b.z,
                      a.w * b.w,
                      a.s4 * b.s4,
                      a.s5 * b.s5,
                      a.s6 * b.s6,
                      a.s7 * b.s7,
                      a.s8 * b.s8,
                      a.s9 * b.s9,
                      a.s10 * b.s10,
                      a.s11 * b.s11,
                      a.s12 * b.s12,
                      a.s13 * b.s13,
                      a.s14 * b.s14,
                      a.s15 * b.s15);
}

occaFunction inline short16  operator *  (const short &a, const short16 &b) {
  return OCCA_SHORT16(a * b.x,
                      a * b.y,
                      a * b.z,
                      a * b.w,
                      a * b.s4,
                      a * b.s5,
                      a * b.s6,
                      a * b.s7,
                      a * b.s8,
                      a * b.s9,
                      a * b.s10,
                      a * b.s11,
                      a * b.s12,
                      a * b.s13,
                      a * b.s14,
                      a * b.s15);
}

occaFunction inline short16  operator *  (const short16 &a, const short &b) {
  return OCCA_SHORT16(a.x * b,
                      a.y * b,
                      a.z * b,
                      a.w * b,
                      a.s4 * b,
                      a.s5 * b,
                      a.s6 * b,
                      a.s7 * b,
                      a.s8 * b,
                      a.s9 * b,
                      a.s10 * b,
                      a.s11 * b,
                      a.s12 * b,
                      a.s13 * b,
                      a.s14 * b,
                      a.s15 * b);
}

occaFunction inline short16& operator *= (      short16 &a, const short16 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  a.s4 *= b.s4;
  a.s5 *= b.s5;
  a.s6 *= b.s6;
  a.s7 *= b.s7;
  a.s8 *= b.s8;
  a.s9 *= b.s9;
  a.s10 *= b.s10;
  a.s11 *= b.s11;
  a.s12 *= b.s12;
  a.s13 *= b.s13;
  a.s14 *= b.s14;
  a.s15 *= b.s15;
  return a;
}

occaFunction inline short16& operator *= (      short16 &a, const short &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  a.s4 *= b;
  a.s5 *= b;
  a.s6 *= b;
  a.s7 *= b;
  a.s8 *= b;
  a.s9 *= b;
  a.s10 *= b;
  a.s11 *= b;
  a.s12 *= b;
  a.s13 *= b;
  a.s14 *= b;
  a.s15 *= b;
  return a;
}
occaFunction inline short16  operator /  (const short16 &a, const short16 &b) {
  return OCCA_SHORT16(a.x / b.x,
                      a.y / b.y,
                      a.z / b.z,
                      a.w / b.w,
                      a.s4 / b.s4,
                      a.s5 / b.s5,
                      a.s6 / b.s6,
                      a.s7 / b.s7,
                      a.s8 / b.s8,
                      a.s9 / b.s9,
                      a.s10 / b.s10,
                      a.s11 / b.s11,
                      a.s12 / b.s12,
                      a.s13 / b.s13,
                      a.s14 / b.s14,
                      a.s15 / b.s15);
}

occaFunction inline short16  operator /  (const short &a, const short16 &b) {
  return OCCA_SHORT16(a / b.x,
                      a / b.y,
                      a / b.z,
                      a / b.w,
                      a / b.s4,
                      a / b.s5,
                      a / b.s6,
                      a / b.s7,
                      a / b.s8,
                      a / b.s9,
                      a / b.s10,
                      a / b.s11,
                      a / b.s12,
                      a / b.s13,
                      a / b.s14,
                      a / b.s15);
}

occaFunction inline short16  operator /  (const short16 &a, const short &b) {
  return OCCA_SHORT16(a.x / b,
                      a.y / b,
                      a.z / b,
                      a.w / b,
                      a.s4 / b,
                      a.s5 / b,
                      a.s6 / b,
                      a.s7 / b,
                      a.s8 / b,
                      a.s9 / b,
                      a.s10 / b,
                      a.s11 / b,
                      a.s12 / b,
                      a.s13 / b,
                      a.s14 / b,
                      a.s15 / b);
}

occaFunction inline short16& operator /= (      short16 &a, const short16 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  a.s4 /= b.s4;
  a.s5 /= b.s5;
  a.s6 /= b.s6;
  a.s7 /= b.s7;
  a.s8 /= b.s8;
  a.s9 /= b.s9;
  a.s10 /= b.s10;
  a.s11 /= b.s11;
  a.s12 /= b.s12;
  a.s13 /= b.s13;
  a.s14 /= b.s14;
  a.s15 /= b.s15;
  return a;
}

occaFunction inline short16& operator /= (      short16 &a, const short &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  a.s4 /= b;
  a.s5 /= b;
  a.s6 /= b;
  a.s7 /= b;
  a.s8 /= b;
  a.s9 /= b;
  a.s10 /= b;
  a.s11 /= b;
  a.s12 /= b;
  a.s13 /= b;
  a.s14 /= b;
  a.s15 /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const short16& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w << ", "
             << a.s4 << ", "
             << a.s5 << ", "
             << a.s6 << ", "
             << a.s7 << ", "
             << a.s8 << ", "
             << a.s9 << ", "
             << a.s10 << ", "
             << a.s11 << ", "
             << a.s12 << ", "
             << a.s13 << ", "
             << a.s14 << ", "
             << a.s15
      << "]\n";

  return out;
}
#endif

//======================================


//---[ int2 ]---------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_INT2 make_int2
#else
#  define OCCA_INT2 int2
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
class int2{
public:
  union { int s0, x; };
  union { int s1, y; };

  inline occaFunction int2() : 
    x(0),
    y(0) {}

  inline occaFunction int2(const int &x_) : 
    x(x_),
    y(0) {}

  inline occaFunction int2(const int &x_,
                           const int &y_) : 
    x(x_),
    y(y_) {}
};
#endif

occaFunction inline int2 operator + (const int2 &a) {
  return OCCA_INT2(+a.x,
                   +a.y);
}

occaFunction inline int2 operator ++ (int2 &a, int) {
  return OCCA_INT2(a.x++,
                   a.y++);
}

occaFunction inline int2& operator ++ (int2 &a) {
  ++a.x;
  ++a.y;
  return a;
}
occaFunction inline int2 operator - (const int2 &a) {
  return OCCA_INT2(-a.x,
                   -a.y);
}

occaFunction inline int2 operator -- (int2 &a, int) {
  return OCCA_INT2(a.x--,
                   a.y--);
}

occaFunction inline int2& operator -- (int2 &a) {
  --a.x;
  --a.y;
  return a;
}
occaFunction inline int2  operator +  (const int2 &a, const int2 &b) {
  return OCCA_INT2(a.x + b.x,
                   a.y + b.y);
}

occaFunction inline int2  operator +  (const int &a, const int2 &b) {
  return OCCA_INT2(a + b.x,
                   a + b.y);
}

occaFunction inline int2  operator +  (const int2 &a, const int &b) {
  return OCCA_INT2(a.x + b,
                   a.y + b);
}

occaFunction inline int2& operator += (      int2 &a, const int2 &b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}

occaFunction inline int2& operator += (      int2 &a, const int &b) {
  a.x += b;
  a.y += b;
  return a;
}
occaFunction inline int2  operator -  (const int2 &a, const int2 &b) {
  return OCCA_INT2(a.x - b.x,
                   a.y - b.y);
}

occaFunction inline int2  operator -  (const int &a, const int2 &b) {
  return OCCA_INT2(a - b.x,
                   a - b.y);
}

occaFunction inline int2  operator -  (const int2 &a, const int &b) {
  return OCCA_INT2(a.x - b,
                   a.y - b);
}

occaFunction inline int2& operator -= (      int2 &a, const int2 &b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

occaFunction inline int2& operator -= (      int2 &a, const int &b) {
  a.x -= b;
  a.y -= b;
  return a;
}
occaFunction inline int2  operator *  (const int2 &a, const int2 &b) {
  return OCCA_INT2(a.x * b.x,
                   a.y * b.y);
}

occaFunction inline int2  operator *  (const int &a, const int2 &b) {
  return OCCA_INT2(a * b.x,
                   a * b.y);
}

occaFunction inline int2  operator *  (const int2 &a, const int &b) {
  return OCCA_INT2(a.x * b,
                   a.y * b);
}

occaFunction inline int2& operator *= (      int2 &a, const int2 &b) {
  a.x *= b.x;
  a.y *= b.y;
  return a;
}

occaFunction inline int2& operator *= (      int2 &a, const int &b) {
  a.x *= b;
  a.y *= b;
  return a;
}
occaFunction inline int2  operator /  (const int2 &a, const int2 &b) {
  return OCCA_INT2(a.x / b.x,
                   a.y / b.y);
}

occaFunction inline int2  operator /  (const int &a, const int2 &b) {
  return OCCA_INT2(a / b.x,
                   a / b.y);
}

occaFunction inline int2  operator /  (const int2 &a, const int &b) {
  return OCCA_INT2(a.x / b,
                   a.y / b);
}

occaFunction inline int2& operator /= (      int2 &a, const int2 &b) {
  a.x /= b.x;
  a.y /= b.y;
  return a;
}

occaFunction inline int2& operator /= (      int2 &a, const int &b) {
  a.x /= b;
  a.y /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const int2& a) {
  out << "[" << a.x << ", "
             << a.y
      << "]\n";

  return out;
}
#endif

//======================================


//---[ int4 ]---------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_INT4 make_int4
#else
#  define OCCA_INT4 int4
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
class int4{
public:
  union { int s0, x; };
  union { int s1, y; };
  union { int s2, z; };
  union { int s3, w; };

  inline occaFunction int4() : 
    x(0),
    y(0),
    z(0),
    w(0) {}

  inline occaFunction int4(const int &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0) {}

  inline occaFunction int4(const int &x_,
                           const int &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0) {}

  inline occaFunction int4(const int &x_,
                           const int &y_,
                           const int &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0) {}

  inline occaFunction int4(const int &x_,
                           const int &y_,
                           const int &z_,
                           const int &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_) {}
};
#endif

occaFunction inline int4 operator + (const int4 &a) {
  return OCCA_INT4(+a.x,
                   +a.y,
                   +a.z,
                   +a.w);
}

occaFunction inline int4 operator ++ (int4 &a, int) {
  return OCCA_INT4(a.x++,
                   a.y++,
                   a.z++,
                   a.w++);
}

occaFunction inline int4& operator ++ (int4 &a) {
  ++a.x;
  ++a.y;
  ++a.z;
  ++a.w;
  return a;
}
occaFunction inline int4 operator - (const int4 &a) {
  return OCCA_INT4(-a.x,
                   -a.y,
                   -a.z,
                   -a.w);
}

occaFunction inline int4 operator -- (int4 &a, int) {
  return OCCA_INT4(a.x--,
                   a.y--,
                   a.z--,
                   a.w--);
}

occaFunction inline int4& operator -- (int4 &a) {
  --a.x;
  --a.y;
  --a.z;
  --a.w;
  return a;
}
occaFunction inline int4  operator +  (const int4 &a, const int4 &b) {
  return OCCA_INT4(a.x + b.x,
                   a.y + b.y,
                   a.z + b.z,
                   a.w + b.w);
}

occaFunction inline int4  operator +  (const int &a, const int4 &b) {
  return OCCA_INT4(a + b.x,
                   a + b.y,
                   a + b.z,
                   a + b.w);
}

occaFunction inline int4  operator +  (const int4 &a, const int &b) {
  return OCCA_INT4(a.x + b,
                   a.y + b,
                   a.z + b,
                   a.w + b);
}

occaFunction inline int4& operator += (      int4 &a, const int4 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

occaFunction inline int4& operator += (      int4 &a, const int &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  return a;
}
occaFunction inline int4  operator -  (const int4 &a, const int4 &b) {
  return OCCA_INT4(a.x - b.x,
                   a.y - b.y,
                   a.z - b.z,
                   a.w - b.w);
}

occaFunction inline int4  operator -  (const int &a, const int4 &b) {
  return OCCA_INT4(a - b.x,
                   a - b.y,
                   a - b.z,
                   a - b.w);
}

occaFunction inline int4  operator -  (const int4 &a, const int &b) {
  return OCCA_INT4(a.x - b,
                   a.y - b,
                   a.z - b,
                   a.w - b);
}

occaFunction inline int4& operator -= (      int4 &a, const int4 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  return a;
}

occaFunction inline int4& operator -= (      int4 &a, const int &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  return a;
}
occaFunction inline int4  operator *  (const int4 &a, const int4 &b) {
  return OCCA_INT4(a.x * b.x,
                   a.y * b.y,
                   a.z * b.z,
                   a.w * b.w);
}

occaFunction inline int4  operator *  (const int &a, const int4 &b) {
  return OCCA_INT4(a * b.x,
                   a * b.y,
                   a * b.z,
                   a * b.w);
}

occaFunction inline int4  operator *  (const int4 &a, const int &b) {
  return OCCA_INT4(a.x * b,
                   a.y * b,
                   a.z * b,
                   a.w * b);
}

occaFunction inline int4& operator *= (      int4 &a, const int4 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  return a;
}

occaFunction inline int4& operator *= (      int4 &a, const int &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}
occaFunction inline int4  operator /  (const int4 &a, const int4 &b) {
  return OCCA_INT4(a.x / b.x,
                   a.y / b.y,
                   a.z / b.z,
                   a.w / b.w);
}

occaFunction inline int4  operator /  (const int &a, const int4 &b) {
  return OCCA_INT4(a / b.x,
                   a / b.y,
                   a / b.z,
                   a / b.w);
}

occaFunction inline int4  operator /  (const int4 &a, const int &b) {
  return OCCA_INT4(a.x / b,
                   a.y / b,
                   a.z / b,
                   a.w / b);
}

occaFunction inline int4& operator /= (      int4 &a, const int4 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  return a;
}

occaFunction inline int4& operator /= (      int4 &a, const int &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const int4& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w
      << "]\n";

  return out;
}
#endif

//======================================


//---[ int3 ]---------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_INT3 make_int3
#else
#  define OCCA_INT3 int3
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
typedef int4 int3;
#endif
//======================================


//---[ int8 ]---------------------------
#define OCCA_INT8 int8
class int8{
public:
  union { int s0, x; };
  union { int s1, y; };
  union { int s2, z; };
  union { int s3, w; };
  int s4;
  int s5;
  int s6;
  int s7;

  inline occaFunction int8() : 
    x(0),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction int8(const int &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction int8(const int &x_,
                           const int &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction int8(const int &x_,
                           const int &y_,
                           const int &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction int8(const int &x_,
                           const int &y_,
                           const int &z_,
                           const int &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction int8(const int &x_,
                           const int &y_,
                           const int &z_,
                           const int &w_,
                           const int &s4_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction int8(const int &x_,
                           const int &y_,
                           const int &z_,
                           const int &w_,
                           const int &s4_,
                           const int &s5_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(0),
    s7(0) {}

  inline occaFunction int8(const int &x_,
                           const int &y_,
                           const int &z_,
                           const int &w_,
                           const int &s4_,
                           const int &s5_,
                           const int &s6_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(0) {}

  inline occaFunction int8(const int &x_,
                           const int &y_,
                           const int &z_,
                           const int &w_,
                           const int &s4_,
                           const int &s5_,
                           const int &s6_,
                           const int &s7_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_) {}
};

occaFunction inline int8 operator + (const int8 &a) {
  return OCCA_INT8(+a.x,
                   +a.y,
                   +a.z,
                   +a.w,
                   +a.s4,
                   +a.s5,
                   +a.s6,
                   +a.s7);
}

occaFunction inline int8 operator ++ (int8 &a, int) {
  return OCCA_INT8(a.x++,
                   a.y++,
                   a.z++,
                   a.w++,
                   a.s4++,
                   a.s5++,
                   a.s6++,
                   a.s7++);
}

occaFunction inline int8& operator ++ (int8 &a) {
  ++a.x;
  ++a.y;
  ++a.z;
  ++a.w;
  ++a.s4;
  ++a.s5;
  ++a.s6;
  ++a.s7;
  return a;
}
occaFunction inline int8 operator - (const int8 &a) {
  return OCCA_INT8(-a.x,
                   -a.y,
                   -a.z,
                   -a.w,
                   -a.s4,
                   -a.s5,
                   -a.s6,
                   -a.s7);
}

occaFunction inline int8 operator -- (int8 &a, int) {
  return OCCA_INT8(a.x--,
                   a.y--,
                   a.z--,
                   a.w--,
                   a.s4--,
                   a.s5--,
                   a.s6--,
                   a.s7--);
}

occaFunction inline int8& operator -- (int8 &a) {
  --a.x;
  --a.y;
  --a.z;
  --a.w;
  --a.s4;
  --a.s5;
  --a.s6;
  --a.s7;
  return a;
}
occaFunction inline int8  operator +  (const int8 &a, const int8 &b) {
  return OCCA_INT8(a.x + b.x,
                   a.y + b.y,
                   a.z + b.z,
                   a.w + b.w,
                   a.s4 + b.s4,
                   a.s5 + b.s5,
                   a.s6 + b.s6,
                   a.s7 + b.s7);
}

occaFunction inline int8  operator +  (const int &a, const int8 &b) {
  return OCCA_INT8(a + b.x,
                   a + b.y,
                   a + b.z,
                   a + b.w,
                   a + b.s4,
                   a + b.s5,
                   a + b.s6,
                   a + b.s7);
}

occaFunction inline int8  operator +  (const int8 &a, const int &b) {
  return OCCA_INT8(a.x + b,
                   a.y + b,
                   a.z + b,
                   a.w + b,
                   a.s4 + b,
                   a.s5 + b,
                   a.s6 + b,
                   a.s7 + b);
}

occaFunction inline int8& operator += (      int8 &a, const int8 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  a.s4 += b.s4;
  a.s5 += b.s5;
  a.s6 += b.s6;
  a.s7 += b.s7;
  return a;
}

occaFunction inline int8& operator += (      int8 &a, const int &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  a.s4 += b;
  a.s5 += b;
  a.s6 += b;
  a.s7 += b;
  return a;
}
occaFunction inline int8  operator -  (const int8 &a, const int8 &b) {
  return OCCA_INT8(a.x - b.x,
                   a.y - b.y,
                   a.z - b.z,
                   a.w - b.w,
                   a.s4 - b.s4,
                   a.s5 - b.s5,
                   a.s6 - b.s6,
                   a.s7 - b.s7);
}

occaFunction inline int8  operator -  (const int &a, const int8 &b) {
  return OCCA_INT8(a - b.x,
                   a - b.y,
                   a - b.z,
                   a - b.w,
                   a - b.s4,
                   a - b.s5,
                   a - b.s6,
                   a - b.s7);
}

occaFunction inline int8  operator -  (const int8 &a, const int &b) {
  return OCCA_INT8(a.x - b,
                   a.y - b,
                   a.z - b,
                   a.w - b,
                   a.s4 - b,
                   a.s5 - b,
                   a.s6 - b,
                   a.s7 - b);
}

occaFunction inline int8& operator -= (      int8 &a, const int8 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  a.s4 -= b.s4;
  a.s5 -= b.s5;
  a.s6 -= b.s6;
  a.s7 -= b.s7;
  return a;
}

occaFunction inline int8& operator -= (      int8 &a, const int &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  a.s4 -= b;
  a.s5 -= b;
  a.s6 -= b;
  a.s7 -= b;
  return a;
}
occaFunction inline int8  operator *  (const int8 &a, const int8 &b) {
  return OCCA_INT8(a.x * b.x,
                   a.y * b.y,
                   a.z * b.z,
                   a.w * b.w,
                   a.s4 * b.s4,
                   a.s5 * b.s5,
                   a.s6 * b.s6,
                   a.s7 * b.s7);
}

occaFunction inline int8  operator *  (const int &a, const int8 &b) {
  return OCCA_INT8(a * b.x,
                   a * b.y,
                   a * b.z,
                   a * b.w,
                   a * b.s4,
                   a * b.s5,
                   a * b.s6,
                   a * b.s7);
}

occaFunction inline int8  operator *  (const int8 &a, const int &b) {
  return OCCA_INT8(a.x * b,
                   a.y * b,
                   a.z * b,
                   a.w * b,
                   a.s4 * b,
                   a.s5 * b,
                   a.s6 * b,
                   a.s7 * b);
}

occaFunction inline int8& operator *= (      int8 &a, const int8 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  a.s4 *= b.s4;
  a.s5 *= b.s5;
  a.s6 *= b.s6;
  a.s7 *= b.s7;
  return a;
}

occaFunction inline int8& operator *= (      int8 &a, const int &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  a.s4 *= b;
  a.s5 *= b;
  a.s6 *= b;
  a.s7 *= b;
  return a;
}
occaFunction inline int8  operator /  (const int8 &a, const int8 &b) {
  return OCCA_INT8(a.x / b.x,
                   a.y / b.y,
                   a.z / b.z,
                   a.w / b.w,
                   a.s4 / b.s4,
                   a.s5 / b.s5,
                   a.s6 / b.s6,
                   a.s7 / b.s7);
}

occaFunction inline int8  operator /  (const int &a, const int8 &b) {
  return OCCA_INT8(a / b.x,
                   a / b.y,
                   a / b.z,
                   a / b.w,
                   a / b.s4,
                   a / b.s5,
                   a / b.s6,
                   a / b.s7);
}

occaFunction inline int8  operator /  (const int8 &a, const int &b) {
  return OCCA_INT8(a.x / b,
                   a.y / b,
                   a.z / b,
                   a.w / b,
                   a.s4 / b,
                   a.s5 / b,
                   a.s6 / b,
                   a.s7 / b);
}

occaFunction inline int8& operator /= (      int8 &a, const int8 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  a.s4 /= b.s4;
  a.s5 /= b.s5;
  a.s6 /= b.s6;
  a.s7 /= b.s7;
  return a;
}

occaFunction inline int8& operator /= (      int8 &a, const int &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  a.s4 /= b;
  a.s5 /= b;
  a.s6 /= b;
  a.s7 /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const int8& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w << ", "
             << a.s4 << ", "
             << a.s5 << ", "
             << a.s6 << ", "
             << a.s7
      << "]\n";

  return out;
}
#endif

//======================================


//---[ int16 ]--------------------------
#define OCCA_INT16 int16
class int16{
public:
  union { int s0, x; };
  union { int s1, y; };
  union { int s2, z; };
  union { int s3, w; };
  int s4;
  int s5;
  int s6;
  int s7;
  int s8;
  int s9;
  int s10;
  int s11;
  int s12;
  int s13;
  int s14;
  int s15;

  inline occaFunction int16() : 
    x(0),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction int16(const int &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction int16(const int &x_,
                            const int &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction int16(const int &x_,
                            const int &y_,
                            const int &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction int16(const int &x_,
                            const int &y_,
                            const int &z_,
                            const int &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction int16(const int &x_,
                            const int &y_,
                            const int &z_,
                            const int &w_,
                            const int &s4_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction int16(const int &x_,
                            const int &y_,
                            const int &z_,
                            const int &w_,
                            const int &s4_,
                            const int &s5_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction int16(const int &x_,
                            const int &y_,
                            const int &z_,
                            const int &w_,
                            const int &s4_,
                            const int &s5_,
                            const int &s6_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction int16(const int &x_,
                            const int &y_,
                            const int &z_,
                            const int &w_,
                            const int &s4_,
                            const int &s5_,
                            const int &s6_,
                            const int &s7_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction int16(const int &x_,
                            const int &y_,
                            const int &z_,
                            const int &w_,
                            const int &s4_,
                            const int &s5_,
                            const int &s6_,
                            const int &s7_,
                            const int &s8_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction int16(const int &x_,
                            const int &y_,
                            const int &z_,
                            const int &w_,
                            const int &s4_,
                            const int &s5_,
                            const int &s6_,
                            const int &s7_,
                            const int &s8_,
                            const int &s9_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction int16(const int &x_,
                            const int &y_,
                            const int &z_,
                            const int &w_,
                            const int &s4_,
                            const int &s5_,
                            const int &s6_,
                            const int &s7_,
                            const int &s8_,
                            const int &s9_,
                            const int &s10_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction int16(const int &x_,
                            const int &y_,
                            const int &z_,
                            const int &w_,
                            const int &s4_,
                            const int &s5_,
                            const int &s6_,
                            const int &s7_,
                            const int &s8_,
                            const int &s9_,
                            const int &s10_,
                            const int &s11_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction int16(const int &x_,
                            const int &y_,
                            const int &z_,
                            const int &w_,
                            const int &s4_,
                            const int &s5_,
                            const int &s6_,
                            const int &s7_,
                            const int &s8_,
                            const int &s9_,
                            const int &s10_,
                            const int &s11_,
                            const int &s12_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction int16(const int &x_,
                            const int &y_,
                            const int &z_,
                            const int &w_,
                            const int &s4_,
                            const int &s5_,
                            const int &s6_,
                            const int &s7_,
                            const int &s8_,
                            const int &s9_,
                            const int &s10_,
                            const int &s11_,
                            const int &s12_,
                            const int &s13_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(0),
    s15(0) {}

  inline occaFunction int16(const int &x_,
                            const int &y_,
                            const int &z_,
                            const int &w_,
                            const int &s4_,
                            const int &s5_,
                            const int &s6_,
                            const int &s7_,
                            const int &s8_,
                            const int &s9_,
                            const int &s10_,
                            const int &s11_,
                            const int &s12_,
                            const int &s13_,
                            const int &s14_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(s14_),
    s15(0) {}

  inline occaFunction int16(const int &x_,
                            const int &y_,
                            const int &z_,
                            const int &w_,
                            const int &s4_,
                            const int &s5_,
                            const int &s6_,
                            const int &s7_,
                            const int &s8_,
                            const int &s9_,
                            const int &s10_,
                            const int &s11_,
                            const int &s12_,
                            const int &s13_,
                            const int &s14_,
                            const int &s15_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(s14_),
    s15(s15_) {}
};

occaFunction inline int16 operator + (const int16 &a) {
  return OCCA_INT16(+a.x,
                    +a.y,
                    +a.z,
                    +a.w,
                    +a.s4,
                    +a.s5,
                    +a.s6,
                    +a.s7,
                    +a.s8,
                    +a.s9,
                    +a.s10,
                    +a.s11,
                    +a.s12,
                    +a.s13,
                    +a.s14,
                    +a.s15);
}

occaFunction inline int16 operator ++ (int16 &a, int) {
  return OCCA_INT16(a.x++,
                    a.y++,
                    a.z++,
                    a.w++,
                    a.s4++,
                    a.s5++,
                    a.s6++,
                    a.s7++,
                    a.s8++,
                    a.s9++,
                    a.s10++,
                    a.s11++,
                    a.s12++,
                    a.s13++,
                    a.s14++,
                    a.s15++);
}

occaFunction inline int16& operator ++ (int16 &a) {
  ++a.x;
  ++a.y;
  ++a.z;
  ++a.w;
  ++a.s4;
  ++a.s5;
  ++a.s6;
  ++a.s7;
  ++a.s8;
  ++a.s9;
  ++a.s10;
  ++a.s11;
  ++a.s12;
  ++a.s13;
  ++a.s14;
  ++a.s15;
  return a;
}
occaFunction inline int16 operator - (const int16 &a) {
  return OCCA_INT16(-a.x,
                    -a.y,
                    -a.z,
                    -a.w,
                    -a.s4,
                    -a.s5,
                    -a.s6,
                    -a.s7,
                    -a.s8,
                    -a.s9,
                    -a.s10,
                    -a.s11,
                    -a.s12,
                    -a.s13,
                    -a.s14,
                    -a.s15);
}

occaFunction inline int16 operator -- (int16 &a, int) {
  return OCCA_INT16(a.x--,
                    a.y--,
                    a.z--,
                    a.w--,
                    a.s4--,
                    a.s5--,
                    a.s6--,
                    a.s7--,
                    a.s8--,
                    a.s9--,
                    a.s10--,
                    a.s11--,
                    a.s12--,
                    a.s13--,
                    a.s14--,
                    a.s15--);
}

occaFunction inline int16& operator -- (int16 &a) {
  --a.x;
  --a.y;
  --a.z;
  --a.w;
  --a.s4;
  --a.s5;
  --a.s6;
  --a.s7;
  --a.s8;
  --a.s9;
  --a.s10;
  --a.s11;
  --a.s12;
  --a.s13;
  --a.s14;
  --a.s15;
  return a;
}
occaFunction inline int16  operator +  (const int16 &a, const int16 &b) {
  return OCCA_INT16(a.x + b.x,
                    a.y + b.y,
                    a.z + b.z,
                    a.w + b.w,
                    a.s4 + b.s4,
                    a.s5 + b.s5,
                    a.s6 + b.s6,
                    a.s7 + b.s7,
                    a.s8 + b.s8,
                    a.s9 + b.s9,
                    a.s10 + b.s10,
                    a.s11 + b.s11,
                    a.s12 + b.s12,
                    a.s13 + b.s13,
                    a.s14 + b.s14,
                    a.s15 + b.s15);
}

occaFunction inline int16  operator +  (const int &a, const int16 &b) {
  return OCCA_INT16(a + b.x,
                    a + b.y,
                    a + b.z,
                    a + b.w,
                    a + b.s4,
                    a + b.s5,
                    a + b.s6,
                    a + b.s7,
                    a + b.s8,
                    a + b.s9,
                    a + b.s10,
                    a + b.s11,
                    a + b.s12,
                    a + b.s13,
                    a + b.s14,
                    a + b.s15);
}

occaFunction inline int16  operator +  (const int16 &a, const int &b) {
  return OCCA_INT16(a.x + b,
                    a.y + b,
                    a.z + b,
                    a.w + b,
                    a.s4 + b,
                    a.s5 + b,
                    a.s6 + b,
                    a.s7 + b,
                    a.s8 + b,
                    a.s9 + b,
                    a.s10 + b,
                    a.s11 + b,
                    a.s12 + b,
                    a.s13 + b,
                    a.s14 + b,
                    a.s15 + b);
}

occaFunction inline int16& operator += (      int16 &a, const int16 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  a.s4 += b.s4;
  a.s5 += b.s5;
  a.s6 += b.s6;
  a.s7 += b.s7;
  a.s8 += b.s8;
  a.s9 += b.s9;
  a.s10 += b.s10;
  a.s11 += b.s11;
  a.s12 += b.s12;
  a.s13 += b.s13;
  a.s14 += b.s14;
  a.s15 += b.s15;
  return a;
}

occaFunction inline int16& operator += (      int16 &a, const int &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  a.s4 += b;
  a.s5 += b;
  a.s6 += b;
  a.s7 += b;
  a.s8 += b;
  a.s9 += b;
  a.s10 += b;
  a.s11 += b;
  a.s12 += b;
  a.s13 += b;
  a.s14 += b;
  a.s15 += b;
  return a;
}
occaFunction inline int16  operator -  (const int16 &a, const int16 &b) {
  return OCCA_INT16(a.x - b.x,
                    a.y - b.y,
                    a.z - b.z,
                    a.w - b.w,
                    a.s4 - b.s4,
                    a.s5 - b.s5,
                    a.s6 - b.s6,
                    a.s7 - b.s7,
                    a.s8 - b.s8,
                    a.s9 - b.s9,
                    a.s10 - b.s10,
                    a.s11 - b.s11,
                    a.s12 - b.s12,
                    a.s13 - b.s13,
                    a.s14 - b.s14,
                    a.s15 - b.s15);
}

occaFunction inline int16  operator -  (const int &a, const int16 &b) {
  return OCCA_INT16(a - b.x,
                    a - b.y,
                    a - b.z,
                    a - b.w,
                    a - b.s4,
                    a - b.s5,
                    a - b.s6,
                    a - b.s7,
                    a - b.s8,
                    a - b.s9,
                    a - b.s10,
                    a - b.s11,
                    a - b.s12,
                    a - b.s13,
                    a - b.s14,
                    a - b.s15);
}

occaFunction inline int16  operator -  (const int16 &a, const int &b) {
  return OCCA_INT16(a.x - b,
                    a.y - b,
                    a.z - b,
                    a.w - b,
                    a.s4 - b,
                    a.s5 - b,
                    a.s6 - b,
                    a.s7 - b,
                    a.s8 - b,
                    a.s9 - b,
                    a.s10 - b,
                    a.s11 - b,
                    a.s12 - b,
                    a.s13 - b,
                    a.s14 - b,
                    a.s15 - b);
}

occaFunction inline int16& operator -= (      int16 &a, const int16 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  a.s4 -= b.s4;
  a.s5 -= b.s5;
  a.s6 -= b.s6;
  a.s7 -= b.s7;
  a.s8 -= b.s8;
  a.s9 -= b.s9;
  a.s10 -= b.s10;
  a.s11 -= b.s11;
  a.s12 -= b.s12;
  a.s13 -= b.s13;
  a.s14 -= b.s14;
  a.s15 -= b.s15;
  return a;
}

occaFunction inline int16& operator -= (      int16 &a, const int &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  a.s4 -= b;
  a.s5 -= b;
  a.s6 -= b;
  a.s7 -= b;
  a.s8 -= b;
  a.s9 -= b;
  a.s10 -= b;
  a.s11 -= b;
  a.s12 -= b;
  a.s13 -= b;
  a.s14 -= b;
  a.s15 -= b;
  return a;
}
occaFunction inline int16  operator *  (const int16 &a, const int16 &b) {
  return OCCA_INT16(a.x * b.x,
                    a.y * b.y,
                    a.z * b.z,
                    a.w * b.w,
                    a.s4 * b.s4,
                    a.s5 * b.s5,
                    a.s6 * b.s6,
                    a.s7 * b.s7,
                    a.s8 * b.s8,
                    a.s9 * b.s9,
                    a.s10 * b.s10,
                    a.s11 * b.s11,
                    a.s12 * b.s12,
                    a.s13 * b.s13,
                    a.s14 * b.s14,
                    a.s15 * b.s15);
}

occaFunction inline int16  operator *  (const int &a, const int16 &b) {
  return OCCA_INT16(a * b.x,
                    a * b.y,
                    a * b.z,
                    a * b.w,
                    a * b.s4,
                    a * b.s5,
                    a * b.s6,
                    a * b.s7,
                    a * b.s8,
                    a * b.s9,
                    a * b.s10,
                    a * b.s11,
                    a * b.s12,
                    a * b.s13,
                    a * b.s14,
                    a * b.s15);
}

occaFunction inline int16  operator *  (const int16 &a, const int &b) {
  return OCCA_INT16(a.x * b,
                    a.y * b,
                    a.z * b,
                    a.w * b,
                    a.s4 * b,
                    a.s5 * b,
                    a.s6 * b,
                    a.s7 * b,
                    a.s8 * b,
                    a.s9 * b,
                    a.s10 * b,
                    a.s11 * b,
                    a.s12 * b,
                    a.s13 * b,
                    a.s14 * b,
                    a.s15 * b);
}

occaFunction inline int16& operator *= (      int16 &a, const int16 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  a.s4 *= b.s4;
  a.s5 *= b.s5;
  a.s6 *= b.s6;
  a.s7 *= b.s7;
  a.s8 *= b.s8;
  a.s9 *= b.s9;
  a.s10 *= b.s10;
  a.s11 *= b.s11;
  a.s12 *= b.s12;
  a.s13 *= b.s13;
  a.s14 *= b.s14;
  a.s15 *= b.s15;
  return a;
}

occaFunction inline int16& operator *= (      int16 &a, const int &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  a.s4 *= b;
  a.s5 *= b;
  a.s6 *= b;
  a.s7 *= b;
  a.s8 *= b;
  a.s9 *= b;
  a.s10 *= b;
  a.s11 *= b;
  a.s12 *= b;
  a.s13 *= b;
  a.s14 *= b;
  a.s15 *= b;
  return a;
}
occaFunction inline int16  operator /  (const int16 &a, const int16 &b) {
  return OCCA_INT16(a.x / b.x,
                    a.y / b.y,
                    a.z / b.z,
                    a.w / b.w,
                    a.s4 / b.s4,
                    a.s5 / b.s5,
                    a.s6 / b.s6,
                    a.s7 / b.s7,
                    a.s8 / b.s8,
                    a.s9 / b.s9,
                    a.s10 / b.s10,
                    a.s11 / b.s11,
                    a.s12 / b.s12,
                    a.s13 / b.s13,
                    a.s14 / b.s14,
                    a.s15 / b.s15);
}

occaFunction inline int16  operator /  (const int &a, const int16 &b) {
  return OCCA_INT16(a / b.x,
                    a / b.y,
                    a / b.z,
                    a / b.w,
                    a / b.s4,
                    a / b.s5,
                    a / b.s6,
                    a / b.s7,
                    a / b.s8,
                    a / b.s9,
                    a / b.s10,
                    a / b.s11,
                    a / b.s12,
                    a / b.s13,
                    a / b.s14,
                    a / b.s15);
}

occaFunction inline int16  operator /  (const int16 &a, const int &b) {
  return OCCA_INT16(a.x / b,
                    a.y / b,
                    a.z / b,
                    a.w / b,
                    a.s4 / b,
                    a.s5 / b,
                    a.s6 / b,
                    a.s7 / b,
                    a.s8 / b,
                    a.s9 / b,
                    a.s10 / b,
                    a.s11 / b,
                    a.s12 / b,
                    a.s13 / b,
                    a.s14 / b,
                    a.s15 / b);
}

occaFunction inline int16& operator /= (      int16 &a, const int16 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  a.s4 /= b.s4;
  a.s5 /= b.s5;
  a.s6 /= b.s6;
  a.s7 /= b.s7;
  a.s8 /= b.s8;
  a.s9 /= b.s9;
  a.s10 /= b.s10;
  a.s11 /= b.s11;
  a.s12 /= b.s12;
  a.s13 /= b.s13;
  a.s14 /= b.s14;
  a.s15 /= b.s15;
  return a;
}

occaFunction inline int16& operator /= (      int16 &a, const int &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  a.s4 /= b;
  a.s5 /= b;
  a.s6 /= b;
  a.s7 /= b;
  a.s8 /= b;
  a.s9 /= b;
  a.s10 /= b;
  a.s11 /= b;
  a.s12 /= b;
  a.s13 /= b;
  a.s14 /= b;
  a.s15 /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const int16& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w << ", "
             << a.s4 << ", "
             << a.s5 << ", "
             << a.s6 << ", "
             << a.s7 << ", "
             << a.s8 << ", "
             << a.s9 << ", "
             << a.s10 << ", "
             << a.s11 << ", "
             << a.s12 << ", "
             << a.s13 << ", "
             << a.s14 << ", "
             << a.s15
      << "]\n";

  return out;
}
#endif

//======================================


//---[ long2 ]--------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_LONG2 make_long2
#else
#  define OCCA_LONG2 long2
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
class long2{
public:
  union { long s0, x; };
  union { long s1, y; };

  inline occaFunction long2() : 
    x(0),
    y(0) {}

  inline occaFunction long2(const long &x_) : 
    x(x_),
    y(0) {}

  inline occaFunction long2(const long &x_,
                            const long &y_) : 
    x(x_),
    y(y_) {}
};
#endif

occaFunction inline long2 operator + (const long2 &a) {
  return OCCA_LONG2(+a.x,
                    +a.y);
}

occaFunction inline long2 operator ++ (long2 &a, int) {
  return OCCA_LONG2(a.x++,
                    a.y++);
}

occaFunction inline long2& operator ++ (long2 &a) {
  ++a.x;
  ++a.y;
  return a;
}
occaFunction inline long2 operator - (const long2 &a) {
  return OCCA_LONG2(-a.x,
                    -a.y);
}

occaFunction inline long2 operator -- (long2 &a, int) {
  return OCCA_LONG2(a.x--,
                    a.y--);
}

occaFunction inline long2& operator -- (long2 &a) {
  --a.x;
  --a.y;
  return a;
}
occaFunction inline long2  operator +  (const long2 &a, const long2 &b) {
  return OCCA_LONG2(a.x + b.x,
                    a.y + b.y);
}

occaFunction inline long2  operator +  (const long &a, const long2 &b) {
  return OCCA_LONG2(a + b.x,
                    a + b.y);
}

occaFunction inline long2  operator +  (const long2 &a, const long &b) {
  return OCCA_LONG2(a.x + b,
                    a.y + b);
}

occaFunction inline long2& operator += (      long2 &a, const long2 &b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}

occaFunction inline long2& operator += (      long2 &a, const long &b) {
  a.x += b;
  a.y += b;
  return a;
}
occaFunction inline long2  operator -  (const long2 &a, const long2 &b) {
  return OCCA_LONG2(a.x - b.x,
                    a.y - b.y);
}

occaFunction inline long2  operator -  (const long &a, const long2 &b) {
  return OCCA_LONG2(a - b.x,
                    a - b.y);
}

occaFunction inline long2  operator -  (const long2 &a, const long &b) {
  return OCCA_LONG2(a.x - b,
                    a.y - b);
}

occaFunction inline long2& operator -= (      long2 &a, const long2 &b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

occaFunction inline long2& operator -= (      long2 &a, const long &b) {
  a.x -= b;
  a.y -= b;
  return a;
}
occaFunction inline long2  operator *  (const long2 &a, const long2 &b) {
  return OCCA_LONG2(a.x * b.x,
                    a.y * b.y);
}

occaFunction inline long2  operator *  (const long &a, const long2 &b) {
  return OCCA_LONG2(a * b.x,
                    a * b.y);
}

occaFunction inline long2  operator *  (const long2 &a, const long &b) {
  return OCCA_LONG2(a.x * b,
                    a.y * b);
}

occaFunction inline long2& operator *= (      long2 &a, const long2 &b) {
  a.x *= b.x;
  a.y *= b.y;
  return a;
}

occaFunction inline long2& operator *= (      long2 &a, const long &b) {
  a.x *= b;
  a.y *= b;
  return a;
}
occaFunction inline long2  operator /  (const long2 &a, const long2 &b) {
  return OCCA_LONG2(a.x / b.x,
                    a.y / b.y);
}

occaFunction inline long2  operator /  (const long &a, const long2 &b) {
  return OCCA_LONG2(a / b.x,
                    a / b.y);
}

occaFunction inline long2  operator /  (const long2 &a, const long &b) {
  return OCCA_LONG2(a.x / b,
                    a.y / b);
}

occaFunction inline long2& operator /= (      long2 &a, const long2 &b) {
  a.x /= b.x;
  a.y /= b.y;
  return a;
}

occaFunction inline long2& operator /= (      long2 &a, const long &b) {
  a.x /= b;
  a.y /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const long2& a) {
  out << "[" << a.x << ", "
             << a.y
      << "]\n";

  return out;
}
#endif

//======================================


//---[ long4 ]--------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_LONG4 make_long4
#else
#  define OCCA_LONG4 long4
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
class long4{
public:
  union { long s0, x; };
  union { long s1, y; };
  union { long s2, z; };
  union { long s3, w; };

  inline occaFunction long4() : 
    x(0),
    y(0),
    z(0),
    w(0) {}

  inline occaFunction long4(const long &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0) {}

  inline occaFunction long4(const long &x_,
                            const long &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0) {}

  inline occaFunction long4(const long &x_,
                            const long &y_,
                            const long &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0) {}

  inline occaFunction long4(const long &x_,
                            const long &y_,
                            const long &z_,
                            const long &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_) {}
};
#endif

occaFunction inline long4 operator + (const long4 &a) {
  return OCCA_LONG4(+a.x,
                    +a.y,
                    +a.z,
                    +a.w);
}

occaFunction inline long4 operator ++ (long4 &a, int) {
  return OCCA_LONG4(a.x++,
                    a.y++,
                    a.z++,
                    a.w++);
}

occaFunction inline long4& operator ++ (long4 &a) {
  ++a.x;
  ++a.y;
  ++a.z;
  ++a.w;
  return a;
}
occaFunction inline long4 operator - (const long4 &a) {
  return OCCA_LONG4(-a.x,
                    -a.y,
                    -a.z,
                    -a.w);
}

occaFunction inline long4 operator -- (long4 &a, int) {
  return OCCA_LONG4(a.x--,
                    a.y--,
                    a.z--,
                    a.w--);
}

occaFunction inline long4& operator -- (long4 &a) {
  --a.x;
  --a.y;
  --a.z;
  --a.w;
  return a;
}
occaFunction inline long4  operator +  (const long4 &a, const long4 &b) {
  return OCCA_LONG4(a.x + b.x,
                    a.y + b.y,
                    a.z + b.z,
                    a.w + b.w);
}

occaFunction inline long4  operator +  (const long &a, const long4 &b) {
  return OCCA_LONG4(a + b.x,
                    a + b.y,
                    a + b.z,
                    a + b.w);
}

occaFunction inline long4  operator +  (const long4 &a, const long &b) {
  return OCCA_LONG4(a.x + b,
                    a.y + b,
                    a.z + b,
                    a.w + b);
}

occaFunction inline long4& operator += (      long4 &a, const long4 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

occaFunction inline long4& operator += (      long4 &a, const long &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  return a;
}
occaFunction inline long4  operator -  (const long4 &a, const long4 &b) {
  return OCCA_LONG4(a.x - b.x,
                    a.y - b.y,
                    a.z - b.z,
                    a.w - b.w);
}

occaFunction inline long4  operator -  (const long &a, const long4 &b) {
  return OCCA_LONG4(a - b.x,
                    a - b.y,
                    a - b.z,
                    a - b.w);
}

occaFunction inline long4  operator -  (const long4 &a, const long &b) {
  return OCCA_LONG4(a.x - b,
                    a.y - b,
                    a.z - b,
                    a.w - b);
}

occaFunction inline long4& operator -= (      long4 &a, const long4 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  return a;
}

occaFunction inline long4& operator -= (      long4 &a, const long &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  return a;
}
occaFunction inline long4  operator *  (const long4 &a, const long4 &b) {
  return OCCA_LONG4(a.x * b.x,
                    a.y * b.y,
                    a.z * b.z,
                    a.w * b.w);
}

occaFunction inline long4  operator *  (const long &a, const long4 &b) {
  return OCCA_LONG4(a * b.x,
                    a * b.y,
                    a * b.z,
                    a * b.w);
}

occaFunction inline long4  operator *  (const long4 &a, const long &b) {
  return OCCA_LONG4(a.x * b,
                    a.y * b,
                    a.z * b,
                    a.w * b);
}

occaFunction inline long4& operator *= (      long4 &a, const long4 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  return a;
}

occaFunction inline long4& operator *= (      long4 &a, const long &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}
occaFunction inline long4  operator /  (const long4 &a, const long4 &b) {
  return OCCA_LONG4(a.x / b.x,
                    a.y / b.y,
                    a.z / b.z,
                    a.w / b.w);
}

occaFunction inline long4  operator /  (const long &a, const long4 &b) {
  return OCCA_LONG4(a / b.x,
                    a / b.y,
                    a / b.z,
                    a / b.w);
}

occaFunction inline long4  operator /  (const long4 &a, const long &b) {
  return OCCA_LONG4(a.x / b,
                    a.y / b,
                    a.z / b,
                    a.w / b);
}

occaFunction inline long4& operator /= (      long4 &a, const long4 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  return a;
}

occaFunction inline long4& operator /= (      long4 &a, const long &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const long4& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w
      << "]\n";

  return out;
}
#endif

//======================================


//---[ long3 ]--------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_LONG3 make_long3
#else
#  define OCCA_LONG3 long3
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
typedef long4 long3;
#endif
//======================================


//---[ long8 ]--------------------------
#define OCCA_LONG8 long8
class long8{
public:
  union { long s0, x; };
  union { long s1, y; };
  union { long s2, z; };
  union { long s3, w; };
  long s4;
  long s5;
  long s6;
  long s7;

  inline occaFunction long8() : 
    x(0),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction long8(const long &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction long8(const long &x_,
                            const long &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction long8(const long &x_,
                            const long &y_,
                            const long &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction long8(const long &x_,
                            const long &y_,
                            const long &z_,
                            const long &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction long8(const long &x_,
                            const long &y_,
                            const long &z_,
                            const long &w_,
                            const long &s4_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction long8(const long &x_,
                            const long &y_,
                            const long &z_,
                            const long &w_,
                            const long &s4_,
                            const long &s5_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(0),
    s7(0) {}

  inline occaFunction long8(const long &x_,
                            const long &y_,
                            const long &z_,
                            const long &w_,
                            const long &s4_,
                            const long &s5_,
                            const long &s6_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(0) {}

  inline occaFunction long8(const long &x_,
                            const long &y_,
                            const long &z_,
                            const long &w_,
                            const long &s4_,
                            const long &s5_,
                            const long &s6_,
                            const long &s7_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_) {}
};

occaFunction inline long8 operator + (const long8 &a) {
  return OCCA_LONG8(+a.x,
                    +a.y,
                    +a.z,
                    +a.w,
                    +a.s4,
                    +a.s5,
                    +a.s6,
                    +a.s7);
}

occaFunction inline long8 operator ++ (long8 &a, int) {
  return OCCA_LONG8(a.x++,
                    a.y++,
                    a.z++,
                    a.w++,
                    a.s4++,
                    a.s5++,
                    a.s6++,
                    a.s7++);
}

occaFunction inline long8& operator ++ (long8 &a) {
  ++a.x;
  ++a.y;
  ++a.z;
  ++a.w;
  ++a.s4;
  ++a.s5;
  ++a.s6;
  ++a.s7;
  return a;
}
occaFunction inline long8 operator - (const long8 &a) {
  return OCCA_LONG8(-a.x,
                    -a.y,
                    -a.z,
                    -a.w,
                    -a.s4,
                    -a.s5,
                    -a.s6,
                    -a.s7);
}

occaFunction inline long8 operator -- (long8 &a, int) {
  return OCCA_LONG8(a.x--,
                    a.y--,
                    a.z--,
                    a.w--,
                    a.s4--,
                    a.s5--,
                    a.s6--,
                    a.s7--);
}

occaFunction inline long8& operator -- (long8 &a) {
  --a.x;
  --a.y;
  --a.z;
  --a.w;
  --a.s4;
  --a.s5;
  --a.s6;
  --a.s7;
  return a;
}
occaFunction inline long8  operator +  (const long8 &a, const long8 &b) {
  return OCCA_LONG8(a.x + b.x,
                    a.y + b.y,
                    a.z + b.z,
                    a.w + b.w,
                    a.s4 + b.s4,
                    a.s5 + b.s5,
                    a.s6 + b.s6,
                    a.s7 + b.s7);
}

occaFunction inline long8  operator +  (const long &a, const long8 &b) {
  return OCCA_LONG8(a + b.x,
                    a + b.y,
                    a + b.z,
                    a + b.w,
                    a + b.s4,
                    a + b.s5,
                    a + b.s6,
                    a + b.s7);
}

occaFunction inline long8  operator +  (const long8 &a, const long &b) {
  return OCCA_LONG8(a.x + b,
                    a.y + b,
                    a.z + b,
                    a.w + b,
                    a.s4 + b,
                    a.s5 + b,
                    a.s6 + b,
                    a.s7 + b);
}

occaFunction inline long8& operator += (      long8 &a, const long8 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  a.s4 += b.s4;
  a.s5 += b.s5;
  a.s6 += b.s6;
  a.s7 += b.s7;
  return a;
}

occaFunction inline long8& operator += (      long8 &a, const long &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  a.s4 += b;
  a.s5 += b;
  a.s6 += b;
  a.s7 += b;
  return a;
}
occaFunction inline long8  operator -  (const long8 &a, const long8 &b) {
  return OCCA_LONG8(a.x - b.x,
                    a.y - b.y,
                    a.z - b.z,
                    a.w - b.w,
                    a.s4 - b.s4,
                    a.s5 - b.s5,
                    a.s6 - b.s6,
                    a.s7 - b.s7);
}

occaFunction inline long8  operator -  (const long &a, const long8 &b) {
  return OCCA_LONG8(a - b.x,
                    a - b.y,
                    a - b.z,
                    a - b.w,
                    a - b.s4,
                    a - b.s5,
                    a - b.s6,
                    a - b.s7);
}

occaFunction inline long8  operator -  (const long8 &a, const long &b) {
  return OCCA_LONG8(a.x - b,
                    a.y - b,
                    a.z - b,
                    a.w - b,
                    a.s4 - b,
                    a.s5 - b,
                    a.s6 - b,
                    a.s7 - b);
}

occaFunction inline long8& operator -= (      long8 &a, const long8 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  a.s4 -= b.s4;
  a.s5 -= b.s5;
  a.s6 -= b.s6;
  a.s7 -= b.s7;
  return a;
}

occaFunction inline long8& operator -= (      long8 &a, const long &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  a.s4 -= b;
  a.s5 -= b;
  a.s6 -= b;
  a.s7 -= b;
  return a;
}
occaFunction inline long8  operator *  (const long8 &a, const long8 &b) {
  return OCCA_LONG8(a.x * b.x,
                    a.y * b.y,
                    a.z * b.z,
                    a.w * b.w,
                    a.s4 * b.s4,
                    a.s5 * b.s5,
                    a.s6 * b.s6,
                    a.s7 * b.s7);
}

occaFunction inline long8  operator *  (const long &a, const long8 &b) {
  return OCCA_LONG8(a * b.x,
                    a * b.y,
                    a * b.z,
                    a * b.w,
                    a * b.s4,
                    a * b.s5,
                    a * b.s6,
                    a * b.s7);
}

occaFunction inline long8  operator *  (const long8 &a, const long &b) {
  return OCCA_LONG8(a.x * b,
                    a.y * b,
                    a.z * b,
                    a.w * b,
                    a.s4 * b,
                    a.s5 * b,
                    a.s6 * b,
                    a.s7 * b);
}

occaFunction inline long8& operator *= (      long8 &a, const long8 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  a.s4 *= b.s4;
  a.s5 *= b.s5;
  a.s6 *= b.s6;
  a.s7 *= b.s7;
  return a;
}

occaFunction inline long8& operator *= (      long8 &a, const long &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  a.s4 *= b;
  a.s5 *= b;
  a.s6 *= b;
  a.s7 *= b;
  return a;
}
occaFunction inline long8  operator /  (const long8 &a, const long8 &b) {
  return OCCA_LONG8(a.x / b.x,
                    a.y / b.y,
                    a.z / b.z,
                    a.w / b.w,
                    a.s4 / b.s4,
                    a.s5 / b.s5,
                    a.s6 / b.s6,
                    a.s7 / b.s7);
}

occaFunction inline long8  operator /  (const long &a, const long8 &b) {
  return OCCA_LONG8(a / b.x,
                    a / b.y,
                    a / b.z,
                    a / b.w,
                    a / b.s4,
                    a / b.s5,
                    a / b.s6,
                    a / b.s7);
}

occaFunction inline long8  operator /  (const long8 &a, const long &b) {
  return OCCA_LONG8(a.x / b,
                    a.y / b,
                    a.z / b,
                    a.w / b,
                    a.s4 / b,
                    a.s5 / b,
                    a.s6 / b,
                    a.s7 / b);
}

occaFunction inline long8& operator /= (      long8 &a, const long8 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  a.s4 /= b.s4;
  a.s5 /= b.s5;
  a.s6 /= b.s6;
  a.s7 /= b.s7;
  return a;
}

occaFunction inline long8& operator /= (      long8 &a, const long &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  a.s4 /= b;
  a.s5 /= b;
  a.s6 /= b;
  a.s7 /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const long8& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w << ", "
             << a.s4 << ", "
             << a.s5 << ", "
             << a.s6 << ", "
             << a.s7
      << "]\n";

  return out;
}
#endif

//======================================


//---[ long16 ]-------------------------
#define OCCA_LONG16 long16
class long16{
public:
  union { long s0, x; };
  union { long s1, y; };
  union { long s2, z; };
  union { long s3, w; };
  long s4;
  long s5;
  long s6;
  long s7;
  long s8;
  long s9;
  long s10;
  long s11;
  long s12;
  long s13;
  long s14;
  long s15;

  inline occaFunction long16() : 
    x(0),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction long16(const long &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction long16(const long &x_,
                             const long &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction long16(const long &x_,
                             const long &y_,
                             const long &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction long16(const long &x_,
                             const long &y_,
                             const long &z_,
                             const long &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction long16(const long &x_,
                             const long &y_,
                             const long &z_,
                             const long &w_,
                             const long &s4_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction long16(const long &x_,
                             const long &y_,
                             const long &z_,
                             const long &w_,
                             const long &s4_,
                             const long &s5_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction long16(const long &x_,
                             const long &y_,
                             const long &z_,
                             const long &w_,
                             const long &s4_,
                             const long &s5_,
                             const long &s6_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction long16(const long &x_,
                             const long &y_,
                             const long &z_,
                             const long &w_,
                             const long &s4_,
                             const long &s5_,
                             const long &s6_,
                             const long &s7_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction long16(const long &x_,
                             const long &y_,
                             const long &z_,
                             const long &w_,
                             const long &s4_,
                             const long &s5_,
                             const long &s6_,
                             const long &s7_,
                             const long &s8_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction long16(const long &x_,
                             const long &y_,
                             const long &z_,
                             const long &w_,
                             const long &s4_,
                             const long &s5_,
                             const long &s6_,
                             const long &s7_,
                             const long &s8_,
                             const long &s9_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction long16(const long &x_,
                             const long &y_,
                             const long &z_,
                             const long &w_,
                             const long &s4_,
                             const long &s5_,
                             const long &s6_,
                             const long &s7_,
                             const long &s8_,
                             const long &s9_,
                             const long &s10_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction long16(const long &x_,
                             const long &y_,
                             const long &z_,
                             const long &w_,
                             const long &s4_,
                             const long &s5_,
                             const long &s6_,
                             const long &s7_,
                             const long &s8_,
                             const long &s9_,
                             const long &s10_,
                             const long &s11_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction long16(const long &x_,
                             const long &y_,
                             const long &z_,
                             const long &w_,
                             const long &s4_,
                             const long &s5_,
                             const long &s6_,
                             const long &s7_,
                             const long &s8_,
                             const long &s9_,
                             const long &s10_,
                             const long &s11_,
                             const long &s12_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction long16(const long &x_,
                             const long &y_,
                             const long &z_,
                             const long &w_,
                             const long &s4_,
                             const long &s5_,
                             const long &s6_,
                             const long &s7_,
                             const long &s8_,
                             const long &s9_,
                             const long &s10_,
                             const long &s11_,
                             const long &s12_,
                             const long &s13_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(0),
    s15(0) {}

  inline occaFunction long16(const long &x_,
                             const long &y_,
                             const long &z_,
                             const long &w_,
                             const long &s4_,
                             const long &s5_,
                             const long &s6_,
                             const long &s7_,
                             const long &s8_,
                             const long &s9_,
                             const long &s10_,
                             const long &s11_,
                             const long &s12_,
                             const long &s13_,
                             const long &s14_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(s14_),
    s15(0) {}

  inline occaFunction long16(const long &x_,
                             const long &y_,
                             const long &z_,
                             const long &w_,
                             const long &s4_,
                             const long &s5_,
                             const long &s6_,
                             const long &s7_,
                             const long &s8_,
                             const long &s9_,
                             const long &s10_,
                             const long &s11_,
                             const long &s12_,
                             const long &s13_,
                             const long &s14_,
                             const long &s15_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(s14_),
    s15(s15_) {}
};

occaFunction inline long16 operator + (const long16 &a) {
  return OCCA_LONG16(+a.x,
                     +a.y,
                     +a.z,
                     +a.w,
                     +a.s4,
                     +a.s5,
                     +a.s6,
                     +a.s7,
                     +a.s8,
                     +a.s9,
                     +a.s10,
                     +a.s11,
                     +a.s12,
                     +a.s13,
                     +a.s14,
                     +a.s15);
}

occaFunction inline long16 operator ++ (long16 &a, int) {
  return OCCA_LONG16(a.x++,
                     a.y++,
                     a.z++,
                     a.w++,
                     a.s4++,
                     a.s5++,
                     a.s6++,
                     a.s7++,
                     a.s8++,
                     a.s9++,
                     a.s10++,
                     a.s11++,
                     a.s12++,
                     a.s13++,
                     a.s14++,
                     a.s15++);
}

occaFunction inline long16& operator ++ (long16 &a) {
  ++a.x;
  ++a.y;
  ++a.z;
  ++a.w;
  ++a.s4;
  ++a.s5;
  ++a.s6;
  ++a.s7;
  ++a.s8;
  ++a.s9;
  ++a.s10;
  ++a.s11;
  ++a.s12;
  ++a.s13;
  ++a.s14;
  ++a.s15;
  return a;
}
occaFunction inline long16 operator - (const long16 &a) {
  return OCCA_LONG16(-a.x,
                     -a.y,
                     -a.z,
                     -a.w,
                     -a.s4,
                     -a.s5,
                     -a.s6,
                     -a.s7,
                     -a.s8,
                     -a.s9,
                     -a.s10,
                     -a.s11,
                     -a.s12,
                     -a.s13,
                     -a.s14,
                     -a.s15);
}

occaFunction inline long16 operator -- (long16 &a, int) {
  return OCCA_LONG16(a.x--,
                     a.y--,
                     a.z--,
                     a.w--,
                     a.s4--,
                     a.s5--,
                     a.s6--,
                     a.s7--,
                     a.s8--,
                     a.s9--,
                     a.s10--,
                     a.s11--,
                     a.s12--,
                     a.s13--,
                     a.s14--,
                     a.s15--);
}

occaFunction inline long16& operator -- (long16 &a) {
  --a.x;
  --a.y;
  --a.z;
  --a.w;
  --a.s4;
  --a.s5;
  --a.s6;
  --a.s7;
  --a.s8;
  --a.s9;
  --a.s10;
  --a.s11;
  --a.s12;
  --a.s13;
  --a.s14;
  --a.s15;
  return a;
}
occaFunction inline long16  operator +  (const long16 &a, const long16 &b) {
  return OCCA_LONG16(a.x + b.x,
                     a.y + b.y,
                     a.z + b.z,
                     a.w + b.w,
                     a.s4 + b.s4,
                     a.s5 + b.s5,
                     a.s6 + b.s6,
                     a.s7 + b.s7,
                     a.s8 + b.s8,
                     a.s9 + b.s9,
                     a.s10 + b.s10,
                     a.s11 + b.s11,
                     a.s12 + b.s12,
                     a.s13 + b.s13,
                     a.s14 + b.s14,
                     a.s15 + b.s15);
}

occaFunction inline long16  operator +  (const long &a, const long16 &b) {
  return OCCA_LONG16(a + b.x,
                     a + b.y,
                     a + b.z,
                     a + b.w,
                     a + b.s4,
                     a + b.s5,
                     a + b.s6,
                     a + b.s7,
                     a + b.s8,
                     a + b.s9,
                     a + b.s10,
                     a + b.s11,
                     a + b.s12,
                     a + b.s13,
                     a + b.s14,
                     a + b.s15);
}

occaFunction inline long16  operator +  (const long16 &a, const long &b) {
  return OCCA_LONG16(a.x + b,
                     a.y + b,
                     a.z + b,
                     a.w + b,
                     a.s4 + b,
                     a.s5 + b,
                     a.s6 + b,
                     a.s7 + b,
                     a.s8 + b,
                     a.s9 + b,
                     a.s10 + b,
                     a.s11 + b,
                     a.s12 + b,
                     a.s13 + b,
                     a.s14 + b,
                     a.s15 + b);
}

occaFunction inline long16& operator += (      long16 &a, const long16 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  a.s4 += b.s4;
  a.s5 += b.s5;
  a.s6 += b.s6;
  a.s7 += b.s7;
  a.s8 += b.s8;
  a.s9 += b.s9;
  a.s10 += b.s10;
  a.s11 += b.s11;
  a.s12 += b.s12;
  a.s13 += b.s13;
  a.s14 += b.s14;
  a.s15 += b.s15;
  return a;
}

occaFunction inline long16& operator += (      long16 &a, const long &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  a.s4 += b;
  a.s5 += b;
  a.s6 += b;
  a.s7 += b;
  a.s8 += b;
  a.s9 += b;
  a.s10 += b;
  a.s11 += b;
  a.s12 += b;
  a.s13 += b;
  a.s14 += b;
  a.s15 += b;
  return a;
}
occaFunction inline long16  operator -  (const long16 &a, const long16 &b) {
  return OCCA_LONG16(a.x - b.x,
                     a.y - b.y,
                     a.z - b.z,
                     a.w - b.w,
                     a.s4 - b.s4,
                     a.s5 - b.s5,
                     a.s6 - b.s6,
                     a.s7 - b.s7,
                     a.s8 - b.s8,
                     a.s9 - b.s9,
                     a.s10 - b.s10,
                     a.s11 - b.s11,
                     a.s12 - b.s12,
                     a.s13 - b.s13,
                     a.s14 - b.s14,
                     a.s15 - b.s15);
}

occaFunction inline long16  operator -  (const long &a, const long16 &b) {
  return OCCA_LONG16(a - b.x,
                     a - b.y,
                     a - b.z,
                     a - b.w,
                     a - b.s4,
                     a - b.s5,
                     a - b.s6,
                     a - b.s7,
                     a - b.s8,
                     a - b.s9,
                     a - b.s10,
                     a - b.s11,
                     a - b.s12,
                     a - b.s13,
                     a - b.s14,
                     a - b.s15);
}

occaFunction inline long16  operator -  (const long16 &a, const long &b) {
  return OCCA_LONG16(a.x - b,
                     a.y - b,
                     a.z - b,
                     a.w - b,
                     a.s4 - b,
                     a.s5 - b,
                     a.s6 - b,
                     a.s7 - b,
                     a.s8 - b,
                     a.s9 - b,
                     a.s10 - b,
                     a.s11 - b,
                     a.s12 - b,
                     a.s13 - b,
                     a.s14 - b,
                     a.s15 - b);
}

occaFunction inline long16& operator -= (      long16 &a, const long16 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  a.s4 -= b.s4;
  a.s5 -= b.s5;
  a.s6 -= b.s6;
  a.s7 -= b.s7;
  a.s8 -= b.s8;
  a.s9 -= b.s9;
  a.s10 -= b.s10;
  a.s11 -= b.s11;
  a.s12 -= b.s12;
  a.s13 -= b.s13;
  a.s14 -= b.s14;
  a.s15 -= b.s15;
  return a;
}

occaFunction inline long16& operator -= (      long16 &a, const long &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  a.s4 -= b;
  a.s5 -= b;
  a.s6 -= b;
  a.s7 -= b;
  a.s8 -= b;
  a.s9 -= b;
  a.s10 -= b;
  a.s11 -= b;
  a.s12 -= b;
  a.s13 -= b;
  a.s14 -= b;
  a.s15 -= b;
  return a;
}
occaFunction inline long16  operator *  (const long16 &a, const long16 &b) {
  return OCCA_LONG16(a.x * b.x,
                     a.y * b.y,
                     a.z * b.z,
                     a.w * b.w,
                     a.s4 * b.s4,
                     a.s5 * b.s5,
                     a.s6 * b.s6,
                     a.s7 * b.s7,
                     a.s8 * b.s8,
                     a.s9 * b.s9,
                     a.s10 * b.s10,
                     a.s11 * b.s11,
                     a.s12 * b.s12,
                     a.s13 * b.s13,
                     a.s14 * b.s14,
                     a.s15 * b.s15);
}

occaFunction inline long16  operator *  (const long &a, const long16 &b) {
  return OCCA_LONG16(a * b.x,
                     a * b.y,
                     a * b.z,
                     a * b.w,
                     a * b.s4,
                     a * b.s5,
                     a * b.s6,
                     a * b.s7,
                     a * b.s8,
                     a * b.s9,
                     a * b.s10,
                     a * b.s11,
                     a * b.s12,
                     a * b.s13,
                     a * b.s14,
                     a * b.s15);
}

occaFunction inline long16  operator *  (const long16 &a, const long &b) {
  return OCCA_LONG16(a.x * b,
                     a.y * b,
                     a.z * b,
                     a.w * b,
                     a.s4 * b,
                     a.s5 * b,
                     a.s6 * b,
                     a.s7 * b,
                     a.s8 * b,
                     a.s9 * b,
                     a.s10 * b,
                     a.s11 * b,
                     a.s12 * b,
                     a.s13 * b,
                     a.s14 * b,
                     a.s15 * b);
}

occaFunction inline long16& operator *= (      long16 &a, const long16 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  a.s4 *= b.s4;
  a.s5 *= b.s5;
  a.s6 *= b.s6;
  a.s7 *= b.s7;
  a.s8 *= b.s8;
  a.s9 *= b.s9;
  a.s10 *= b.s10;
  a.s11 *= b.s11;
  a.s12 *= b.s12;
  a.s13 *= b.s13;
  a.s14 *= b.s14;
  a.s15 *= b.s15;
  return a;
}

occaFunction inline long16& operator *= (      long16 &a, const long &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  a.s4 *= b;
  a.s5 *= b;
  a.s6 *= b;
  a.s7 *= b;
  a.s8 *= b;
  a.s9 *= b;
  a.s10 *= b;
  a.s11 *= b;
  a.s12 *= b;
  a.s13 *= b;
  a.s14 *= b;
  a.s15 *= b;
  return a;
}
occaFunction inline long16  operator /  (const long16 &a, const long16 &b) {
  return OCCA_LONG16(a.x / b.x,
                     a.y / b.y,
                     a.z / b.z,
                     a.w / b.w,
                     a.s4 / b.s4,
                     a.s5 / b.s5,
                     a.s6 / b.s6,
                     a.s7 / b.s7,
                     a.s8 / b.s8,
                     a.s9 / b.s9,
                     a.s10 / b.s10,
                     a.s11 / b.s11,
                     a.s12 / b.s12,
                     a.s13 / b.s13,
                     a.s14 / b.s14,
                     a.s15 / b.s15);
}

occaFunction inline long16  operator /  (const long &a, const long16 &b) {
  return OCCA_LONG16(a / b.x,
                     a / b.y,
                     a / b.z,
                     a / b.w,
                     a / b.s4,
                     a / b.s5,
                     a / b.s6,
                     a / b.s7,
                     a / b.s8,
                     a / b.s9,
                     a / b.s10,
                     a / b.s11,
                     a / b.s12,
                     a / b.s13,
                     a / b.s14,
                     a / b.s15);
}

occaFunction inline long16  operator /  (const long16 &a, const long &b) {
  return OCCA_LONG16(a.x / b,
                     a.y / b,
                     a.z / b,
                     a.w / b,
                     a.s4 / b,
                     a.s5 / b,
                     a.s6 / b,
                     a.s7 / b,
                     a.s8 / b,
                     a.s9 / b,
                     a.s10 / b,
                     a.s11 / b,
                     a.s12 / b,
                     a.s13 / b,
                     a.s14 / b,
                     a.s15 / b);
}

occaFunction inline long16& operator /= (      long16 &a, const long16 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  a.s4 /= b.s4;
  a.s5 /= b.s5;
  a.s6 /= b.s6;
  a.s7 /= b.s7;
  a.s8 /= b.s8;
  a.s9 /= b.s9;
  a.s10 /= b.s10;
  a.s11 /= b.s11;
  a.s12 /= b.s12;
  a.s13 /= b.s13;
  a.s14 /= b.s14;
  a.s15 /= b.s15;
  return a;
}

occaFunction inline long16& operator /= (      long16 &a, const long &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  a.s4 /= b;
  a.s5 /= b;
  a.s6 /= b;
  a.s7 /= b;
  a.s8 /= b;
  a.s9 /= b;
  a.s10 /= b;
  a.s11 /= b;
  a.s12 /= b;
  a.s13 /= b;
  a.s14 /= b;
  a.s15 /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const long16& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w << ", "
             << a.s4 << ", "
             << a.s5 << ", "
             << a.s6 << ", "
             << a.s7 << ", "
             << a.s8 << ", "
             << a.s9 << ", "
             << a.s10 << ", "
             << a.s11 << ", "
             << a.s12 << ", "
             << a.s13 << ", "
             << a.s14 << ", "
             << a.s15
      << "]\n";

  return out;
}
#endif

//======================================


//---[ float2 ]-------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_FLOAT2 make_float2
#else
#  define OCCA_FLOAT2 float2
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
class float2{
public:
  union { float s0, x; };
  union { float s1, y; };

  inline occaFunction float2() : 
    x(0),
    y(0) {}

  inline occaFunction float2(const float &x_) : 
    x(x_),
    y(0) {}

  inline occaFunction float2(const float &x_,
                             const float &y_) : 
    x(x_),
    y(y_) {}
};
#endif

occaFunction inline float2 operator + (const float2 &a) {
  return OCCA_FLOAT2(+a.x,
                     +a.y);
}
occaFunction inline float2 operator - (const float2 &a) {
  return OCCA_FLOAT2(-a.x,
                     -a.y);
}
occaFunction inline float2  operator +  (const float2 &a, const float2 &b) {
  return OCCA_FLOAT2(a.x + b.x,
                     a.y + b.y);
}

occaFunction inline float2  operator +  (const float &a, const float2 &b) {
  return OCCA_FLOAT2(a + b.x,
                     a + b.y);
}

occaFunction inline float2  operator +  (const float2 &a, const float &b) {
  return OCCA_FLOAT2(a.x + b,
                     a.y + b);
}

occaFunction inline float2& operator += (      float2 &a, const float2 &b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}

occaFunction inline float2& operator += (      float2 &a, const float &b) {
  a.x += b;
  a.y += b;
  return a;
}
occaFunction inline float2  operator -  (const float2 &a, const float2 &b) {
  return OCCA_FLOAT2(a.x - b.x,
                     a.y - b.y);
}

occaFunction inline float2  operator -  (const float &a, const float2 &b) {
  return OCCA_FLOAT2(a - b.x,
                     a - b.y);
}

occaFunction inline float2  operator -  (const float2 &a, const float &b) {
  return OCCA_FLOAT2(a.x - b,
                     a.y - b);
}

occaFunction inline float2& operator -= (      float2 &a, const float2 &b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

occaFunction inline float2& operator -= (      float2 &a, const float &b) {
  a.x -= b;
  a.y -= b;
  return a;
}
occaFunction inline float2  operator *  (const float2 &a, const float2 &b) {
  return OCCA_FLOAT2(a.x * b.x,
                     a.y * b.y);
}

occaFunction inline float2  operator *  (const float &a, const float2 &b) {
  return OCCA_FLOAT2(a * b.x,
                     a * b.y);
}

occaFunction inline float2  operator *  (const float2 &a, const float &b) {
  return OCCA_FLOAT2(a.x * b,
                     a.y * b);
}

occaFunction inline float2& operator *= (      float2 &a, const float2 &b) {
  a.x *= b.x;
  a.y *= b.y;
  return a;
}

occaFunction inline float2& operator *= (      float2 &a, const float &b) {
  a.x *= b;
  a.y *= b;
  return a;
}
occaFunction inline float2  operator /  (const float2 &a, const float2 &b) {
  return OCCA_FLOAT2(a.x / b.x,
                     a.y / b.y);
}

occaFunction inline float2  operator /  (const float &a, const float2 &b) {
  return OCCA_FLOAT2(a / b.x,
                     a / b.y);
}

occaFunction inline float2  operator /  (const float2 &a, const float &b) {
  return OCCA_FLOAT2(a.x / b,
                     a.y / b);
}

occaFunction inline float2& operator /= (      float2 &a, const float2 &b) {
  a.x /= b.x;
  a.y /= b.y;
  return a;
}

occaFunction inline float2& operator /= (      float2 &a, const float &b) {
  a.x /= b;
  a.y /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const float2& a) {
  out << "[" << a.x << ", "
             << a.y
      << "]\n";

  return out;
}
#endif

//======================================


//---[ float4 ]-------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_FLOAT4 make_float4
#else
#  define OCCA_FLOAT4 float4
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
class float4{
public:
  union { float s0, x; };
  union { float s1, y; };
  union { float s2, z; };
  union { float s3, w; };

  inline occaFunction float4() : 
    x(0),
    y(0),
    z(0),
    w(0) {}

  inline occaFunction float4(const float &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0) {}

  inline occaFunction float4(const float &x_,
                             const float &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0) {}

  inline occaFunction float4(const float &x_,
                             const float &y_,
                             const float &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0) {}

  inline occaFunction float4(const float &x_,
                             const float &y_,
                             const float &z_,
                             const float &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_) {}
};
#endif

occaFunction inline float4 operator + (const float4 &a) {
  return OCCA_FLOAT4(+a.x,
                     +a.y,
                     +a.z,
                     +a.w);
}
occaFunction inline float4 operator - (const float4 &a) {
  return OCCA_FLOAT4(-a.x,
                     -a.y,
                     -a.z,
                     -a.w);
}
occaFunction inline float4  operator +  (const float4 &a, const float4 &b) {
  return OCCA_FLOAT4(a.x + b.x,
                     a.y + b.y,
                     a.z + b.z,
                     a.w + b.w);
}

occaFunction inline float4  operator +  (const float &a, const float4 &b) {
  return OCCA_FLOAT4(a + b.x,
                     a + b.y,
                     a + b.z,
                     a + b.w);
}

occaFunction inline float4  operator +  (const float4 &a, const float &b) {
  return OCCA_FLOAT4(a.x + b,
                     a.y + b,
                     a.z + b,
                     a.w + b);
}

occaFunction inline float4& operator += (      float4 &a, const float4 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

occaFunction inline float4& operator += (      float4 &a, const float &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  return a;
}
occaFunction inline float4  operator -  (const float4 &a, const float4 &b) {
  return OCCA_FLOAT4(a.x - b.x,
                     a.y - b.y,
                     a.z - b.z,
                     a.w - b.w);
}

occaFunction inline float4  operator -  (const float &a, const float4 &b) {
  return OCCA_FLOAT4(a - b.x,
                     a - b.y,
                     a - b.z,
                     a - b.w);
}

occaFunction inline float4  operator -  (const float4 &a, const float &b) {
  return OCCA_FLOAT4(a.x - b,
                     a.y - b,
                     a.z - b,
                     a.w - b);
}

occaFunction inline float4& operator -= (      float4 &a, const float4 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  return a;
}

occaFunction inline float4& operator -= (      float4 &a, const float &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  return a;
}
occaFunction inline float4  operator *  (const float4 &a, const float4 &b) {
  return OCCA_FLOAT4(a.x * b.x,
                     a.y * b.y,
                     a.z * b.z,
                     a.w * b.w);
}

occaFunction inline float4  operator *  (const float &a, const float4 &b) {
  return OCCA_FLOAT4(a * b.x,
                     a * b.y,
                     a * b.z,
                     a * b.w);
}

occaFunction inline float4  operator *  (const float4 &a, const float &b) {
  return OCCA_FLOAT4(a.x * b,
                     a.y * b,
                     a.z * b,
                     a.w * b);
}

occaFunction inline float4& operator *= (      float4 &a, const float4 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  return a;
}

occaFunction inline float4& operator *= (      float4 &a, const float &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}
occaFunction inline float4  operator /  (const float4 &a, const float4 &b) {
  return OCCA_FLOAT4(a.x / b.x,
                     a.y / b.y,
                     a.z / b.z,
                     a.w / b.w);
}

occaFunction inline float4  operator /  (const float &a, const float4 &b) {
  return OCCA_FLOAT4(a / b.x,
                     a / b.y,
                     a / b.z,
                     a / b.w);
}

occaFunction inline float4  operator /  (const float4 &a, const float &b) {
  return OCCA_FLOAT4(a.x / b,
                     a.y / b,
                     a.z / b,
                     a.w / b);
}

occaFunction inline float4& operator /= (      float4 &a, const float4 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  return a;
}

occaFunction inline float4& operator /= (      float4 &a, const float &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const float4& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w
      << "]\n";

  return out;
}
#endif

//======================================


//---[ float3 ]-------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_FLOAT3 make_float3
#else
#  define OCCA_FLOAT3 float3
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
typedef float4 float3;
#endif
//======================================


//---[ float8 ]-------------------------
#define OCCA_FLOAT8 float8
class float8{
public:
  union { float s0, x; };
  union { float s1, y; };
  union { float s2, z; };
  union { float s3, w; };
  float s4;
  float s5;
  float s6;
  float s7;

  inline occaFunction float8() : 
    x(0),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction float8(const float &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction float8(const float &x_,
                             const float &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction float8(const float &x_,
                             const float &y_,
                             const float &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction float8(const float &x_,
                             const float &y_,
                             const float &z_,
                             const float &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction float8(const float &x_,
                             const float &y_,
                             const float &z_,
                             const float &w_,
                             const float &s4_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction float8(const float &x_,
                             const float &y_,
                             const float &z_,
                             const float &w_,
                             const float &s4_,
                             const float &s5_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(0),
    s7(0) {}

  inline occaFunction float8(const float &x_,
                             const float &y_,
                             const float &z_,
                             const float &w_,
                             const float &s4_,
                             const float &s5_,
                             const float &s6_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(0) {}

  inline occaFunction float8(const float &x_,
                             const float &y_,
                             const float &z_,
                             const float &w_,
                             const float &s4_,
                             const float &s5_,
                             const float &s6_,
                             const float &s7_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_) {}
};

occaFunction inline float8 operator + (const float8 &a) {
  return OCCA_FLOAT8(+a.x,
                     +a.y,
                     +a.z,
                     +a.w,
                     +a.s4,
                     +a.s5,
                     +a.s6,
                     +a.s7);
}
occaFunction inline float8 operator - (const float8 &a) {
  return OCCA_FLOAT8(-a.x,
                     -a.y,
                     -a.z,
                     -a.w,
                     -a.s4,
                     -a.s5,
                     -a.s6,
                     -a.s7);
}
occaFunction inline float8  operator +  (const float8 &a, const float8 &b) {
  return OCCA_FLOAT8(a.x + b.x,
                     a.y + b.y,
                     a.z + b.z,
                     a.w + b.w,
                     a.s4 + b.s4,
                     a.s5 + b.s5,
                     a.s6 + b.s6,
                     a.s7 + b.s7);
}

occaFunction inline float8  operator +  (const float &a, const float8 &b) {
  return OCCA_FLOAT8(a + b.x,
                     a + b.y,
                     a + b.z,
                     a + b.w,
                     a + b.s4,
                     a + b.s5,
                     a + b.s6,
                     a + b.s7);
}

occaFunction inline float8  operator +  (const float8 &a, const float &b) {
  return OCCA_FLOAT8(a.x + b,
                     a.y + b,
                     a.z + b,
                     a.w + b,
                     a.s4 + b,
                     a.s5 + b,
                     a.s6 + b,
                     a.s7 + b);
}

occaFunction inline float8& operator += (      float8 &a, const float8 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  a.s4 += b.s4;
  a.s5 += b.s5;
  a.s6 += b.s6;
  a.s7 += b.s7;
  return a;
}

occaFunction inline float8& operator += (      float8 &a, const float &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  a.s4 += b;
  a.s5 += b;
  a.s6 += b;
  a.s7 += b;
  return a;
}
occaFunction inline float8  operator -  (const float8 &a, const float8 &b) {
  return OCCA_FLOAT8(a.x - b.x,
                     a.y - b.y,
                     a.z - b.z,
                     a.w - b.w,
                     a.s4 - b.s4,
                     a.s5 - b.s5,
                     a.s6 - b.s6,
                     a.s7 - b.s7);
}

occaFunction inline float8  operator -  (const float &a, const float8 &b) {
  return OCCA_FLOAT8(a - b.x,
                     a - b.y,
                     a - b.z,
                     a - b.w,
                     a - b.s4,
                     a - b.s5,
                     a - b.s6,
                     a - b.s7);
}

occaFunction inline float8  operator -  (const float8 &a, const float &b) {
  return OCCA_FLOAT8(a.x - b,
                     a.y - b,
                     a.z - b,
                     a.w - b,
                     a.s4 - b,
                     a.s5 - b,
                     a.s6 - b,
                     a.s7 - b);
}

occaFunction inline float8& operator -= (      float8 &a, const float8 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  a.s4 -= b.s4;
  a.s5 -= b.s5;
  a.s6 -= b.s6;
  a.s7 -= b.s7;
  return a;
}

occaFunction inline float8& operator -= (      float8 &a, const float &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  a.s4 -= b;
  a.s5 -= b;
  a.s6 -= b;
  a.s7 -= b;
  return a;
}
occaFunction inline float8  operator *  (const float8 &a, const float8 &b) {
  return OCCA_FLOAT8(a.x * b.x,
                     a.y * b.y,
                     a.z * b.z,
                     a.w * b.w,
                     a.s4 * b.s4,
                     a.s5 * b.s5,
                     a.s6 * b.s6,
                     a.s7 * b.s7);
}

occaFunction inline float8  operator *  (const float &a, const float8 &b) {
  return OCCA_FLOAT8(a * b.x,
                     a * b.y,
                     a * b.z,
                     a * b.w,
                     a * b.s4,
                     a * b.s5,
                     a * b.s6,
                     a * b.s7);
}

occaFunction inline float8  operator *  (const float8 &a, const float &b) {
  return OCCA_FLOAT8(a.x * b,
                     a.y * b,
                     a.z * b,
                     a.w * b,
                     a.s4 * b,
                     a.s5 * b,
                     a.s6 * b,
                     a.s7 * b);
}

occaFunction inline float8& operator *= (      float8 &a, const float8 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  a.s4 *= b.s4;
  a.s5 *= b.s5;
  a.s6 *= b.s6;
  a.s7 *= b.s7;
  return a;
}

occaFunction inline float8& operator *= (      float8 &a, const float &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  a.s4 *= b;
  a.s5 *= b;
  a.s6 *= b;
  a.s7 *= b;
  return a;
}
occaFunction inline float8  operator /  (const float8 &a, const float8 &b) {
  return OCCA_FLOAT8(a.x / b.x,
                     a.y / b.y,
                     a.z / b.z,
                     a.w / b.w,
                     a.s4 / b.s4,
                     a.s5 / b.s5,
                     a.s6 / b.s6,
                     a.s7 / b.s7);
}

occaFunction inline float8  operator /  (const float &a, const float8 &b) {
  return OCCA_FLOAT8(a / b.x,
                     a / b.y,
                     a / b.z,
                     a / b.w,
                     a / b.s4,
                     a / b.s5,
                     a / b.s6,
                     a / b.s7);
}

occaFunction inline float8  operator /  (const float8 &a, const float &b) {
  return OCCA_FLOAT8(a.x / b,
                     a.y / b,
                     a.z / b,
                     a.w / b,
                     a.s4 / b,
                     a.s5 / b,
                     a.s6 / b,
                     a.s7 / b);
}

occaFunction inline float8& operator /= (      float8 &a, const float8 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  a.s4 /= b.s4;
  a.s5 /= b.s5;
  a.s6 /= b.s6;
  a.s7 /= b.s7;
  return a;
}

occaFunction inline float8& operator /= (      float8 &a, const float &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  a.s4 /= b;
  a.s5 /= b;
  a.s6 /= b;
  a.s7 /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const float8& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w << ", "
             << a.s4 << ", "
             << a.s5 << ", "
             << a.s6 << ", "
             << a.s7
      << "]\n";

  return out;
}
#endif

//======================================


//---[ float16 ]------------------------
#define OCCA_FLOAT16 float16
class float16{
public:
  union { float s0, x; };
  union { float s1, y; };
  union { float s2, z; };
  union { float s3, w; };
  float s4;
  float s5;
  float s6;
  float s7;
  float s8;
  float s9;
  float s10;
  float s11;
  float s12;
  float s13;
  float s14;
  float s15;

  inline occaFunction float16() : 
    x(0),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction float16(const float &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction float16(const float &x_,
                              const float &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction float16(const float &x_,
                              const float &y_,
                              const float &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction float16(const float &x_,
                              const float &y_,
                              const float &z_,
                              const float &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction float16(const float &x_,
                              const float &y_,
                              const float &z_,
                              const float &w_,
                              const float &s4_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction float16(const float &x_,
                              const float &y_,
                              const float &z_,
                              const float &w_,
                              const float &s4_,
                              const float &s5_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction float16(const float &x_,
                              const float &y_,
                              const float &z_,
                              const float &w_,
                              const float &s4_,
                              const float &s5_,
                              const float &s6_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction float16(const float &x_,
                              const float &y_,
                              const float &z_,
                              const float &w_,
                              const float &s4_,
                              const float &s5_,
                              const float &s6_,
                              const float &s7_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction float16(const float &x_,
                              const float &y_,
                              const float &z_,
                              const float &w_,
                              const float &s4_,
                              const float &s5_,
                              const float &s6_,
                              const float &s7_,
                              const float &s8_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction float16(const float &x_,
                              const float &y_,
                              const float &z_,
                              const float &w_,
                              const float &s4_,
                              const float &s5_,
                              const float &s6_,
                              const float &s7_,
                              const float &s8_,
                              const float &s9_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction float16(const float &x_,
                              const float &y_,
                              const float &z_,
                              const float &w_,
                              const float &s4_,
                              const float &s5_,
                              const float &s6_,
                              const float &s7_,
                              const float &s8_,
                              const float &s9_,
                              const float &s10_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction float16(const float &x_,
                              const float &y_,
                              const float &z_,
                              const float &w_,
                              const float &s4_,
                              const float &s5_,
                              const float &s6_,
                              const float &s7_,
                              const float &s8_,
                              const float &s9_,
                              const float &s10_,
                              const float &s11_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction float16(const float &x_,
                              const float &y_,
                              const float &z_,
                              const float &w_,
                              const float &s4_,
                              const float &s5_,
                              const float &s6_,
                              const float &s7_,
                              const float &s8_,
                              const float &s9_,
                              const float &s10_,
                              const float &s11_,
                              const float &s12_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction float16(const float &x_,
                              const float &y_,
                              const float &z_,
                              const float &w_,
                              const float &s4_,
                              const float &s5_,
                              const float &s6_,
                              const float &s7_,
                              const float &s8_,
                              const float &s9_,
                              const float &s10_,
                              const float &s11_,
                              const float &s12_,
                              const float &s13_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(0),
    s15(0) {}

  inline occaFunction float16(const float &x_,
                              const float &y_,
                              const float &z_,
                              const float &w_,
                              const float &s4_,
                              const float &s5_,
                              const float &s6_,
                              const float &s7_,
                              const float &s8_,
                              const float &s9_,
                              const float &s10_,
                              const float &s11_,
                              const float &s12_,
                              const float &s13_,
                              const float &s14_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(s14_),
    s15(0) {}

  inline occaFunction float16(const float &x_,
                              const float &y_,
                              const float &z_,
                              const float &w_,
                              const float &s4_,
                              const float &s5_,
                              const float &s6_,
                              const float &s7_,
                              const float &s8_,
                              const float &s9_,
                              const float &s10_,
                              const float &s11_,
                              const float &s12_,
                              const float &s13_,
                              const float &s14_,
                              const float &s15_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(s14_),
    s15(s15_) {}
};

occaFunction inline float16 operator + (const float16 &a) {
  return OCCA_FLOAT16(+a.x,
                      +a.y,
                      +a.z,
                      +a.w,
                      +a.s4,
                      +a.s5,
                      +a.s6,
                      +a.s7,
                      +a.s8,
                      +a.s9,
                      +a.s10,
                      +a.s11,
                      +a.s12,
                      +a.s13,
                      +a.s14,
                      +a.s15);
}
occaFunction inline float16 operator - (const float16 &a) {
  return OCCA_FLOAT16(-a.x,
                      -a.y,
                      -a.z,
                      -a.w,
                      -a.s4,
                      -a.s5,
                      -a.s6,
                      -a.s7,
                      -a.s8,
                      -a.s9,
                      -a.s10,
                      -a.s11,
                      -a.s12,
                      -a.s13,
                      -a.s14,
                      -a.s15);
}
occaFunction inline float16  operator +  (const float16 &a, const float16 &b) {
  return OCCA_FLOAT16(a.x + b.x,
                      a.y + b.y,
                      a.z + b.z,
                      a.w + b.w,
                      a.s4 + b.s4,
                      a.s5 + b.s5,
                      a.s6 + b.s6,
                      a.s7 + b.s7,
                      a.s8 + b.s8,
                      a.s9 + b.s9,
                      a.s10 + b.s10,
                      a.s11 + b.s11,
                      a.s12 + b.s12,
                      a.s13 + b.s13,
                      a.s14 + b.s14,
                      a.s15 + b.s15);
}

occaFunction inline float16  operator +  (const float &a, const float16 &b) {
  return OCCA_FLOAT16(a + b.x,
                      a + b.y,
                      a + b.z,
                      a + b.w,
                      a + b.s4,
                      a + b.s5,
                      a + b.s6,
                      a + b.s7,
                      a + b.s8,
                      a + b.s9,
                      a + b.s10,
                      a + b.s11,
                      a + b.s12,
                      a + b.s13,
                      a + b.s14,
                      a + b.s15);
}

occaFunction inline float16  operator +  (const float16 &a, const float &b) {
  return OCCA_FLOAT16(a.x + b,
                      a.y + b,
                      a.z + b,
                      a.w + b,
                      a.s4 + b,
                      a.s5 + b,
                      a.s6 + b,
                      a.s7 + b,
                      a.s8 + b,
                      a.s9 + b,
                      a.s10 + b,
                      a.s11 + b,
                      a.s12 + b,
                      a.s13 + b,
                      a.s14 + b,
                      a.s15 + b);
}

occaFunction inline float16& operator += (      float16 &a, const float16 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  a.s4 += b.s4;
  a.s5 += b.s5;
  a.s6 += b.s6;
  a.s7 += b.s7;
  a.s8 += b.s8;
  a.s9 += b.s9;
  a.s10 += b.s10;
  a.s11 += b.s11;
  a.s12 += b.s12;
  a.s13 += b.s13;
  a.s14 += b.s14;
  a.s15 += b.s15;
  return a;
}

occaFunction inline float16& operator += (      float16 &a, const float &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  a.s4 += b;
  a.s5 += b;
  a.s6 += b;
  a.s7 += b;
  a.s8 += b;
  a.s9 += b;
  a.s10 += b;
  a.s11 += b;
  a.s12 += b;
  a.s13 += b;
  a.s14 += b;
  a.s15 += b;
  return a;
}
occaFunction inline float16  operator -  (const float16 &a, const float16 &b) {
  return OCCA_FLOAT16(a.x - b.x,
                      a.y - b.y,
                      a.z - b.z,
                      a.w - b.w,
                      a.s4 - b.s4,
                      a.s5 - b.s5,
                      a.s6 - b.s6,
                      a.s7 - b.s7,
                      a.s8 - b.s8,
                      a.s9 - b.s9,
                      a.s10 - b.s10,
                      a.s11 - b.s11,
                      a.s12 - b.s12,
                      a.s13 - b.s13,
                      a.s14 - b.s14,
                      a.s15 - b.s15);
}

occaFunction inline float16  operator -  (const float &a, const float16 &b) {
  return OCCA_FLOAT16(a - b.x,
                      a - b.y,
                      a - b.z,
                      a - b.w,
                      a - b.s4,
                      a - b.s5,
                      a - b.s6,
                      a - b.s7,
                      a - b.s8,
                      a - b.s9,
                      a - b.s10,
                      a - b.s11,
                      a - b.s12,
                      a - b.s13,
                      a - b.s14,
                      a - b.s15);
}

occaFunction inline float16  operator -  (const float16 &a, const float &b) {
  return OCCA_FLOAT16(a.x - b,
                      a.y - b,
                      a.z - b,
                      a.w - b,
                      a.s4 - b,
                      a.s5 - b,
                      a.s6 - b,
                      a.s7 - b,
                      a.s8 - b,
                      a.s9 - b,
                      a.s10 - b,
                      a.s11 - b,
                      a.s12 - b,
                      a.s13 - b,
                      a.s14 - b,
                      a.s15 - b);
}

occaFunction inline float16& operator -= (      float16 &a, const float16 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  a.s4 -= b.s4;
  a.s5 -= b.s5;
  a.s6 -= b.s6;
  a.s7 -= b.s7;
  a.s8 -= b.s8;
  a.s9 -= b.s9;
  a.s10 -= b.s10;
  a.s11 -= b.s11;
  a.s12 -= b.s12;
  a.s13 -= b.s13;
  a.s14 -= b.s14;
  a.s15 -= b.s15;
  return a;
}

occaFunction inline float16& operator -= (      float16 &a, const float &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  a.s4 -= b;
  a.s5 -= b;
  a.s6 -= b;
  a.s7 -= b;
  a.s8 -= b;
  a.s9 -= b;
  a.s10 -= b;
  a.s11 -= b;
  a.s12 -= b;
  a.s13 -= b;
  a.s14 -= b;
  a.s15 -= b;
  return a;
}
occaFunction inline float16  operator *  (const float16 &a, const float16 &b) {
  return OCCA_FLOAT16(a.x * b.x,
                      a.y * b.y,
                      a.z * b.z,
                      a.w * b.w,
                      a.s4 * b.s4,
                      a.s5 * b.s5,
                      a.s6 * b.s6,
                      a.s7 * b.s7,
                      a.s8 * b.s8,
                      a.s9 * b.s9,
                      a.s10 * b.s10,
                      a.s11 * b.s11,
                      a.s12 * b.s12,
                      a.s13 * b.s13,
                      a.s14 * b.s14,
                      a.s15 * b.s15);
}

occaFunction inline float16  operator *  (const float &a, const float16 &b) {
  return OCCA_FLOAT16(a * b.x,
                      a * b.y,
                      a * b.z,
                      a * b.w,
                      a * b.s4,
                      a * b.s5,
                      a * b.s6,
                      a * b.s7,
                      a * b.s8,
                      a * b.s9,
                      a * b.s10,
                      a * b.s11,
                      a * b.s12,
                      a * b.s13,
                      a * b.s14,
                      a * b.s15);
}

occaFunction inline float16  operator *  (const float16 &a, const float &b) {
  return OCCA_FLOAT16(a.x * b,
                      a.y * b,
                      a.z * b,
                      a.w * b,
                      a.s4 * b,
                      a.s5 * b,
                      a.s6 * b,
                      a.s7 * b,
                      a.s8 * b,
                      a.s9 * b,
                      a.s10 * b,
                      a.s11 * b,
                      a.s12 * b,
                      a.s13 * b,
                      a.s14 * b,
                      a.s15 * b);
}

occaFunction inline float16& operator *= (      float16 &a, const float16 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  a.s4 *= b.s4;
  a.s5 *= b.s5;
  a.s6 *= b.s6;
  a.s7 *= b.s7;
  a.s8 *= b.s8;
  a.s9 *= b.s9;
  a.s10 *= b.s10;
  a.s11 *= b.s11;
  a.s12 *= b.s12;
  a.s13 *= b.s13;
  a.s14 *= b.s14;
  a.s15 *= b.s15;
  return a;
}

occaFunction inline float16& operator *= (      float16 &a, const float &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  a.s4 *= b;
  a.s5 *= b;
  a.s6 *= b;
  a.s7 *= b;
  a.s8 *= b;
  a.s9 *= b;
  a.s10 *= b;
  a.s11 *= b;
  a.s12 *= b;
  a.s13 *= b;
  a.s14 *= b;
  a.s15 *= b;
  return a;
}
occaFunction inline float16  operator /  (const float16 &a, const float16 &b) {
  return OCCA_FLOAT16(a.x / b.x,
                      a.y / b.y,
                      a.z / b.z,
                      a.w / b.w,
                      a.s4 / b.s4,
                      a.s5 / b.s5,
                      a.s6 / b.s6,
                      a.s7 / b.s7,
                      a.s8 / b.s8,
                      a.s9 / b.s9,
                      a.s10 / b.s10,
                      a.s11 / b.s11,
                      a.s12 / b.s12,
                      a.s13 / b.s13,
                      a.s14 / b.s14,
                      a.s15 / b.s15);
}

occaFunction inline float16  operator /  (const float &a, const float16 &b) {
  return OCCA_FLOAT16(a / b.x,
                      a / b.y,
                      a / b.z,
                      a / b.w,
                      a / b.s4,
                      a / b.s5,
                      a / b.s6,
                      a / b.s7,
                      a / b.s8,
                      a / b.s9,
                      a / b.s10,
                      a / b.s11,
                      a / b.s12,
                      a / b.s13,
                      a / b.s14,
                      a / b.s15);
}

occaFunction inline float16  operator /  (const float16 &a, const float &b) {
  return OCCA_FLOAT16(a.x / b,
                      a.y / b,
                      a.z / b,
                      a.w / b,
                      a.s4 / b,
                      a.s5 / b,
                      a.s6 / b,
                      a.s7 / b,
                      a.s8 / b,
                      a.s9 / b,
                      a.s10 / b,
                      a.s11 / b,
                      a.s12 / b,
                      a.s13 / b,
                      a.s14 / b,
                      a.s15 / b);
}

occaFunction inline float16& operator /= (      float16 &a, const float16 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  a.s4 /= b.s4;
  a.s5 /= b.s5;
  a.s6 /= b.s6;
  a.s7 /= b.s7;
  a.s8 /= b.s8;
  a.s9 /= b.s9;
  a.s10 /= b.s10;
  a.s11 /= b.s11;
  a.s12 /= b.s12;
  a.s13 /= b.s13;
  a.s14 /= b.s14;
  a.s15 /= b.s15;
  return a;
}

occaFunction inline float16& operator /= (      float16 &a, const float &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  a.s4 /= b;
  a.s5 /= b;
  a.s6 /= b;
  a.s7 /= b;
  a.s8 /= b;
  a.s9 /= b;
  a.s10 /= b;
  a.s11 /= b;
  a.s12 /= b;
  a.s13 /= b;
  a.s14 /= b;
  a.s15 /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const float16& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w << ", "
             << a.s4 << ", "
             << a.s5 << ", "
             << a.s6 << ", "
             << a.s7 << ", "
             << a.s8 << ", "
             << a.s9 << ", "
             << a.s10 << ", "
             << a.s11 << ", "
             << a.s12 << ", "
             << a.s13 << ", "
             << a.s14 << ", "
             << a.s15
      << "]\n";

  return out;
}
#endif

//======================================


//---[ double2 ]------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_DOUBLE2 make_double2
#else
#  define OCCA_DOUBLE2 double2
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
class double2{
public:
  union { double s0, x; };
  union { double s1, y; };

  inline occaFunction double2() : 
    x(0),
    y(0) {}

  inline occaFunction double2(const double &x_) : 
    x(x_),
    y(0) {}

  inline occaFunction double2(const double &x_,
                              const double &y_) : 
    x(x_),
    y(y_) {}
};
#endif

occaFunction inline double2 operator + (const double2 &a) {
  return OCCA_DOUBLE2(+a.x,
                      +a.y);
}
occaFunction inline double2 operator - (const double2 &a) {
  return OCCA_DOUBLE2(-a.x,
                      -a.y);
}
occaFunction inline double2  operator +  (const double2 &a, const double2 &b) {
  return OCCA_DOUBLE2(a.x + b.x,
                      a.y + b.y);
}

occaFunction inline double2  operator +  (const double &a, const double2 &b) {
  return OCCA_DOUBLE2(a + b.x,
                      a + b.y);
}

occaFunction inline double2  operator +  (const double2 &a, const double &b) {
  return OCCA_DOUBLE2(a.x + b,
                      a.y + b);
}

occaFunction inline double2& operator += (      double2 &a, const double2 &b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}

occaFunction inline double2& operator += (      double2 &a, const double &b) {
  a.x += b;
  a.y += b;
  return a;
}
occaFunction inline double2  operator -  (const double2 &a, const double2 &b) {
  return OCCA_DOUBLE2(a.x - b.x,
                      a.y - b.y);
}

occaFunction inline double2  operator -  (const double &a, const double2 &b) {
  return OCCA_DOUBLE2(a - b.x,
                      a - b.y);
}

occaFunction inline double2  operator -  (const double2 &a, const double &b) {
  return OCCA_DOUBLE2(a.x - b,
                      a.y - b);
}

occaFunction inline double2& operator -= (      double2 &a, const double2 &b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

occaFunction inline double2& operator -= (      double2 &a, const double &b) {
  a.x -= b;
  a.y -= b;
  return a;
}
occaFunction inline double2  operator *  (const double2 &a, const double2 &b) {
  return OCCA_DOUBLE2(a.x * b.x,
                      a.y * b.y);
}

occaFunction inline double2  operator *  (const double &a, const double2 &b) {
  return OCCA_DOUBLE2(a * b.x,
                      a * b.y);
}

occaFunction inline double2  operator *  (const double2 &a, const double &b) {
  return OCCA_DOUBLE2(a.x * b,
                      a.y * b);
}

occaFunction inline double2& operator *= (      double2 &a, const double2 &b) {
  a.x *= b.x;
  a.y *= b.y;
  return a;
}

occaFunction inline double2& operator *= (      double2 &a, const double &b) {
  a.x *= b;
  a.y *= b;
  return a;
}
occaFunction inline double2  operator /  (const double2 &a, const double2 &b) {
  return OCCA_DOUBLE2(a.x / b.x,
                      a.y / b.y);
}

occaFunction inline double2  operator /  (const double &a, const double2 &b) {
  return OCCA_DOUBLE2(a / b.x,
                      a / b.y);
}

occaFunction inline double2  operator /  (const double2 &a, const double &b) {
  return OCCA_DOUBLE2(a.x / b,
                      a.y / b);
}

occaFunction inline double2& operator /= (      double2 &a, const double2 &b) {
  a.x /= b.x;
  a.y /= b.y;
  return a;
}

occaFunction inline double2& operator /= (      double2 &a, const double &b) {
  a.x /= b;
  a.y /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const double2& a) {
  out << "[" << a.x << ", "
             << a.y
      << "]\n";

  return out;
}
#endif

//======================================


//---[ double4 ]------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_DOUBLE4 make_double4
#else
#  define OCCA_DOUBLE4 double4
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
class double4{
public:
  union { double s0, x; };
  union { double s1, y; };
  union { double s2, z; };
  union { double s3, w; };

  inline occaFunction double4() : 
    x(0),
    y(0),
    z(0),
    w(0) {}

  inline occaFunction double4(const double &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0) {}

  inline occaFunction double4(const double &x_,
                              const double &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0) {}

  inline occaFunction double4(const double &x_,
                              const double &y_,
                              const double &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0) {}

  inline occaFunction double4(const double &x_,
                              const double &y_,
                              const double &z_,
                              const double &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_) {}
};
#endif

occaFunction inline double4 operator + (const double4 &a) {
  return OCCA_DOUBLE4(+a.x,
                      +a.y,
                      +a.z,
                      +a.w);
}
occaFunction inline double4 operator - (const double4 &a) {
  return OCCA_DOUBLE4(-a.x,
                      -a.y,
                      -a.z,
                      -a.w);
}
occaFunction inline double4  operator +  (const double4 &a, const double4 &b) {
  return OCCA_DOUBLE4(a.x + b.x,
                      a.y + b.y,
                      a.z + b.z,
                      a.w + b.w);
}

occaFunction inline double4  operator +  (const double &a, const double4 &b) {
  return OCCA_DOUBLE4(a + b.x,
                      a + b.y,
                      a + b.z,
                      a + b.w);
}

occaFunction inline double4  operator +  (const double4 &a, const double &b) {
  return OCCA_DOUBLE4(a.x + b,
                      a.y + b,
                      a.z + b,
                      a.w + b);
}

occaFunction inline double4& operator += (      double4 &a, const double4 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

occaFunction inline double4& operator += (      double4 &a, const double &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  return a;
}
occaFunction inline double4  operator -  (const double4 &a, const double4 &b) {
  return OCCA_DOUBLE4(a.x - b.x,
                      a.y - b.y,
                      a.z - b.z,
                      a.w - b.w);
}

occaFunction inline double4  operator -  (const double &a, const double4 &b) {
  return OCCA_DOUBLE4(a - b.x,
                      a - b.y,
                      a - b.z,
                      a - b.w);
}

occaFunction inline double4  operator -  (const double4 &a, const double &b) {
  return OCCA_DOUBLE4(a.x - b,
                      a.y - b,
                      a.z - b,
                      a.w - b);
}

occaFunction inline double4& operator -= (      double4 &a, const double4 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  return a;
}

occaFunction inline double4& operator -= (      double4 &a, const double &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  return a;
}
occaFunction inline double4  operator *  (const double4 &a, const double4 &b) {
  return OCCA_DOUBLE4(a.x * b.x,
                      a.y * b.y,
                      a.z * b.z,
                      a.w * b.w);
}

occaFunction inline double4  operator *  (const double &a, const double4 &b) {
  return OCCA_DOUBLE4(a * b.x,
                      a * b.y,
                      a * b.z,
                      a * b.w);
}

occaFunction inline double4  operator *  (const double4 &a, const double &b) {
  return OCCA_DOUBLE4(a.x * b,
                      a.y * b,
                      a.z * b,
                      a.w * b);
}

occaFunction inline double4& operator *= (      double4 &a, const double4 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  return a;
}

occaFunction inline double4& operator *= (      double4 &a, const double &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}
occaFunction inline double4  operator /  (const double4 &a, const double4 &b) {
  return OCCA_DOUBLE4(a.x / b.x,
                      a.y / b.y,
                      a.z / b.z,
                      a.w / b.w);
}

occaFunction inline double4  operator /  (const double &a, const double4 &b) {
  return OCCA_DOUBLE4(a / b.x,
                      a / b.y,
                      a / b.z,
                      a / b.w);
}

occaFunction inline double4  operator /  (const double4 &a, const double &b) {
  return OCCA_DOUBLE4(a.x / b,
                      a.y / b,
                      a.z / b,
                      a.w / b);
}

occaFunction inline double4& operator /= (      double4 &a, const double4 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  return a;
}

occaFunction inline double4& operator /= (      double4 &a, const double &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const double4& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w
      << "]\n";

  return out;
}
#endif

//======================================


//---[ double3 ]------------------------
#if (defined(OCCA_IN_KERNEL) && OCCA_USING_CUDA)
#  define OCCA_DOUBLE3 make_double3
#else
#  define OCCA_DOUBLE3 double3
#endif
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
typedef double4 double3;
#endif
//======================================


//---[ double8 ]------------------------
#define OCCA_DOUBLE8 double8
class double8{
public:
  union { double s0, x; };
  union { double s1, y; };
  union { double s2, z; };
  union { double s3, w; };
  double s4;
  double s5;
  double s6;
  double s7;

  inline occaFunction double8() : 
    x(0),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction double8(const double &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction double8(const double &x_,
                              const double &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction double8(const double &x_,
                              const double &y_,
                              const double &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction double8(const double &x_,
                              const double &y_,
                              const double &z_,
                              const double &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(0),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction double8(const double &x_,
                              const double &y_,
                              const double &z_,
                              const double &w_,
                              const double &s4_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(0),
    s6(0),
    s7(0) {}

  inline occaFunction double8(const double &x_,
                              const double &y_,
                              const double &z_,
                              const double &w_,
                              const double &s4_,
                              const double &s5_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(0),
    s7(0) {}

  inline occaFunction double8(const double &x_,
                              const double &y_,
                              const double &z_,
                              const double &w_,
                              const double &s4_,
                              const double &s5_,
                              const double &s6_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(0) {}

  inline occaFunction double8(const double &x_,
                              const double &y_,
                              const double &z_,
                              const double &w_,
                              const double &s4_,
                              const double &s5_,
                              const double &s6_,
                              const double &s7_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_) {}
};

occaFunction inline double8 operator + (const double8 &a) {
  return OCCA_DOUBLE8(+a.x,
                      +a.y,
                      +a.z,
                      +a.w,
                      +a.s4,
                      +a.s5,
                      +a.s6,
                      +a.s7);
}
occaFunction inline double8 operator - (const double8 &a) {
  return OCCA_DOUBLE8(-a.x,
                      -a.y,
                      -a.z,
                      -a.w,
                      -a.s4,
                      -a.s5,
                      -a.s6,
                      -a.s7);
}
occaFunction inline double8  operator +  (const double8 &a, const double8 &b) {
  return OCCA_DOUBLE8(a.x + b.x,
                      a.y + b.y,
                      a.z + b.z,
                      a.w + b.w,
                      a.s4 + b.s4,
                      a.s5 + b.s5,
                      a.s6 + b.s6,
                      a.s7 + b.s7);
}

occaFunction inline double8  operator +  (const double &a, const double8 &b) {
  return OCCA_DOUBLE8(a + b.x,
                      a + b.y,
                      a + b.z,
                      a + b.w,
                      a + b.s4,
                      a + b.s5,
                      a + b.s6,
                      a + b.s7);
}

occaFunction inline double8  operator +  (const double8 &a, const double &b) {
  return OCCA_DOUBLE8(a.x + b,
                      a.y + b,
                      a.z + b,
                      a.w + b,
                      a.s4 + b,
                      a.s5 + b,
                      a.s6 + b,
                      a.s7 + b);
}

occaFunction inline double8& operator += (      double8 &a, const double8 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  a.s4 += b.s4;
  a.s5 += b.s5;
  a.s6 += b.s6;
  a.s7 += b.s7;
  return a;
}

occaFunction inline double8& operator += (      double8 &a, const double &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  a.s4 += b;
  a.s5 += b;
  a.s6 += b;
  a.s7 += b;
  return a;
}
occaFunction inline double8  operator -  (const double8 &a, const double8 &b) {
  return OCCA_DOUBLE8(a.x - b.x,
                      a.y - b.y,
                      a.z - b.z,
                      a.w - b.w,
                      a.s4 - b.s4,
                      a.s5 - b.s5,
                      a.s6 - b.s6,
                      a.s7 - b.s7);
}

occaFunction inline double8  operator -  (const double &a, const double8 &b) {
  return OCCA_DOUBLE8(a - b.x,
                      a - b.y,
                      a - b.z,
                      a - b.w,
                      a - b.s4,
                      a - b.s5,
                      a - b.s6,
                      a - b.s7);
}

occaFunction inline double8  operator -  (const double8 &a, const double &b) {
  return OCCA_DOUBLE8(a.x - b,
                      a.y - b,
                      a.z - b,
                      a.w - b,
                      a.s4 - b,
                      a.s5 - b,
                      a.s6 - b,
                      a.s7 - b);
}

occaFunction inline double8& operator -= (      double8 &a, const double8 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  a.s4 -= b.s4;
  a.s5 -= b.s5;
  a.s6 -= b.s6;
  a.s7 -= b.s7;
  return a;
}

occaFunction inline double8& operator -= (      double8 &a, const double &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  a.s4 -= b;
  a.s5 -= b;
  a.s6 -= b;
  a.s7 -= b;
  return a;
}
occaFunction inline double8  operator *  (const double8 &a, const double8 &b) {
  return OCCA_DOUBLE8(a.x * b.x,
                      a.y * b.y,
                      a.z * b.z,
                      a.w * b.w,
                      a.s4 * b.s4,
                      a.s5 * b.s5,
                      a.s6 * b.s6,
                      a.s7 * b.s7);
}

occaFunction inline double8  operator *  (const double &a, const double8 &b) {
  return OCCA_DOUBLE8(a * b.x,
                      a * b.y,
                      a * b.z,
                      a * b.w,
                      a * b.s4,
                      a * b.s5,
                      a * b.s6,
                      a * b.s7);
}

occaFunction inline double8  operator *  (const double8 &a, const double &b) {
  return OCCA_DOUBLE8(a.x * b,
                      a.y * b,
                      a.z * b,
                      a.w * b,
                      a.s4 * b,
                      a.s5 * b,
                      a.s6 * b,
                      a.s7 * b);
}

occaFunction inline double8& operator *= (      double8 &a, const double8 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  a.s4 *= b.s4;
  a.s5 *= b.s5;
  a.s6 *= b.s6;
  a.s7 *= b.s7;
  return a;
}

occaFunction inline double8& operator *= (      double8 &a, const double &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  a.s4 *= b;
  a.s5 *= b;
  a.s6 *= b;
  a.s7 *= b;
  return a;
}
occaFunction inline double8  operator /  (const double8 &a, const double8 &b) {
  return OCCA_DOUBLE8(a.x / b.x,
                      a.y / b.y,
                      a.z / b.z,
                      a.w / b.w,
                      a.s4 / b.s4,
                      a.s5 / b.s5,
                      a.s6 / b.s6,
                      a.s7 / b.s7);
}

occaFunction inline double8  operator /  (const double &a, const double8 &b) {
  return OCCA_DOUBLE8(a / b.x,
                      a / b.y,
                      a / b.z,
                      a / b.w,
                      a / b.s4,
                      a / b.s5,
                      a / b.s6,
                      a / b.s7);
}

occaFunction inline double8  operator /  (const double8 &a, const double &b) {
  return OCCA_DOUBLE8(a.x / b,
                      a.y / b,
                      a.z / b,
                      a.w / b,
                      a.s4 / b,
                      a.s5 / b,
                      a.s6 / b,
                      a.s7 / b);
}

occaFunction inline double8& operator /= (      double8 &a, const double8 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  a.s4 /= b.s4;
  a.s5 /= b.s5;
  a.s6 /= b.s6;
  a.s7 /= b.s7;
  return a;
}

occaFunction inline double8& operator /= (      double8 &a, const double &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  a.s4 /= b;
  a.s5 /= b;
  a.s6 /= b;
  a.s7 /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const double8& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w << ", "
             << a.s4 << ", "
             << a.s5 << ", "
             << a.s6 << ", "
             << a.s7
      << "]\n";

  return out;
}
#endif

//======================================


//---[ double16 ]-----------------------
#define OCCA_DOUBLE16 double16
class double16{
public:
  union { double s0, x; };
  union { double s1, y; };
  union { double s2, z; };
  union { double s3, w; };
  double s4;
  double s5;
  double s6;
  double s7;
  double s8;
  double s9;
  double s10;
  double s11;
  double s12;
  double s13;
  double s14;
  double s15;

  inline occaFunction double16() : 
    x(0),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction double16(const double &x_) : 
    x(x_),
    y(0),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction double16(const double &x_,
                               const double &y_) : 
    x(x_),
    y(y_),
    z(0),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction double16(const double &x_,
                               const double &y_,
                               const double &z_) : 
    x(x_),
    y(y_),
    z(z_),
    w(0),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction double16(const double &x_,
                               const double &y_,
                               const double &z_,
                               const double &w_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(0),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction double16(const double &x_,
                               const double &y_,
                               const double &z_,
                               const double &w_,
                               const double &s4_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(0),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction double16(const double &x_,
                               const double &y_,
                               const double &z_,
                               const double &w_,
                               const double &s4_,
                               const double &s5_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(0),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction double16(const double &x_,
                               const double &y_,
                               const double &z_,
                               const double &w_,
                               const double &s4_,
                               const double &s5_,
                               const double &s6_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(0),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction double16(const double &x_,
                               const double &y_,
                               const double &z_,
                               const double &w_,
                               const double &s4_,
                               const double &s5_,
                               const double &s6_,
                               const double &s7_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(0),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction double16(const double &x_,
                               const double &y_,
                               const double &z_,
                               const double &w_,
                               const double &s4_,
                               const double &s5_,
                               const double &s6_,
                               const double &s7_,
                               const double &s8_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(0),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction double16(const double &x_,
                               const double &y_,
                               const double &z_,
                               const double &w_,
                               const double &s4_,
                               const double &s5_,
                               const double &s6_,
                               const double &s7_,
                               const double &s8_,
                               const double &s9_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(0),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction double16(const double &x_,
                               const double &y_,
                               const double &z_,
                               const double &w_,
                               const double &s4_,
                               const double &s5_,
                               const double &s6_,
                               const double &s7_,
                               const double &s8_,
                               const double &s9_,
                               const double &s10_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(0),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction double16(const double &x_,
                               const double &y_,
                               const double &z_,
                               const double &w_,
                               const double &s4_,
                               const double &s5_,
                               const double &s6_,
                               const double &s7_,
                               const double &s8_,
                               const double &s9_,
                               const double &s10_,
                               const double &s11_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(0),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction double16(const double &x_,
                               const double &y_,
                               const double &z_,
                               const double &w_,
                               const double &s4_,
                               const double &s5_,
                               const double &s6_,
                               const double &s7_,
                               const double &s8_,
                               const double &s9_,
                               const double &s10_,
                               const double &s11_,
                               const double &s12_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(0),
    s14(0),
    s15(0) {}

  inline occaFunction double16(const double &x_,
                               const double &y_,
                               const double &z_,
                               const double &w_,
                               const double &s4_,
                               const double &s5_,
                               const double &s6_,
                               const double &s7_,
                               const double &s8_,
                               const double &s9_,
                               const double &s10_,
                               const double &s11_,
                               const double &s12_,
                               const double &s13_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(0),
    s15(0) {}

  inline occaFunction double16(const double &x_,
                               const double &y_,
                               const double &z_,
                               const double &w_,
                               const double &s4_,
                               const double &s5_,
                               const double &s6_,
                               const double &s7_,
                               const double &s8_,
                               const double &s9_,
                               const double &s10_,
                               const double &s11_,
                               const double &s12_,
                               const double &s13_,
                               const double &s14_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(s14_),
    s15(0) {}

  inline occaFunction double16(const double &x_,
                               const double &y_,
                               const double &z_,
                               const double &w_,
                               const double &s4_,
                               const double &s5_,
                               const double &s6_,
                               const double &s7_,
                               const double &s8_,
                               const double &s9_,
                               const double &s10_,
                               const double &s11_,
                               const double &s12_,
                               const double &s13_,
                               const double &s14_,
                               const double &s15_) : 
    x(x_),
    y(y_),
    z(z_),
    w(w_),
    s4(s4_),
    s5(s5_),
    s6(s6_),
    s7(s7_),
    s8(s8_),
    s9(s9_),
    s10(s10_),
    s11(s11_),
    s12(s12_),
    s13(s13_),
    s14(s14_),
    s15(s15_) {}
};

occaFunction inline double16 operator + (const double16 &a) {
  return OCCA_DOUBLE16(+a.x,
                       +a.y,
                       +a.z,
                       +a.w,
                       +a.s4,
                       +a.s5,
                       +a.s6,
                       +a.s7,
                       +a.s8,
                       +a.s9,
                       +a.s10,
                       +a.s11,
                       +a.s12,
                       +a.s13,
                       +a.s14,
                       +a.s15);
}
occaFunction inline double16 operator - (const double16 &a) {
  return OCCA_DOUBLE16(-a.x,
                       -a.y,
                       -a.z,
                       -a.w,
                       -a.s4,
                       -a.s5,
                       -a.s6,
                       -a.s7,
                       -a.s8,
                       -a.s9,
                       -a.s10,
                       -a.s11,
                       -a.s12,
                       -a.s13,
                       -a.s14,
                       -a.s15);
}
occaFunction inline double16  operator +  (const double16 &a, const double16 &b) {
  return OCCA_DOUBLE16(a.x + b.x,
                       a.y + b.y,
                       a.z + b.z,
                       a.w + b.w,
                       a.s4 + b.s4,
                       a.s5 + b.s5,
                       a.s6 + b.s6,
                       a.s7 + b.s7,
                       a.s8 + b.s8,
                       a.s9 + b.s9,
                       a.s10 + b.s10,
                       a.s11 + b.s11,
                       a.s12 + b.s12,
                       a.s13 + b.s13,
                       a.s14 + b.s14,
                       a.s15 + b.s15);
}

occaFunction inline double16  operator +  (const double &a, const double16 &b) {
  return OCCA_DOUBLE16(a + b.x,
                       a + b.y,
                       a + b.z,
                       a + b.w,
                       a + b.s4,
                       a + b.s5,
                       a + b.s6,
                       a + b.s7,
                       a + b.s8,
                       a + b.s9,
                       a + b.s10,
                       a + b.s11,
                       a + b.s12,
                       a + b.s13,
                       a + b.s14,
                       a + b.s15);
}

occaFunction inline double16  operator +  (const double16 &a, const double &b) {
  return OCCA_DOUBLE16(a.x + b,
                       a.y + b,
                       a.z + b,
                       a.w + b,
                       a.s4 + b,
                       a.s5 + b,
                       a.s6 + b,
                       a.s7 + b,
                       a.s8 + b,
                       a.s9 + b,
                       a.s10 + b,
                       a.s11 + b,
                       a.s12 + b,
                       a.s13 + b,
                       a.s14 + b,
                       a.s15 + b);
}

occaFunction inline double16& operator += (      double16 &a, const double16 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  a.s4 += b.s4;
  a.s5 += b.s5;
  a.s6 += b.s6;
  a.s7 += b.s7;
  a.s8 += b.s8;
  a.s9 += b.s9;
  a.s10 += b.s10;
  a.s11 += b.s11;
  a.s12 += b.s12;
  a.s13 += b.s13;
  a.s14 += b.s14;
  a.s15 += b.s15;
  return a;
}

occaFunction inline double16& operator += (      double16 &a, const double &b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  a.s4 += b;
  a.s5 += b;
  a.s6 += b;
  a.s7 += b;
  a.s8 += b;
  a.s9 += b;
  a.s10 += b;
  a.s11 += b;
  a.s12 += b;
  a.s13 += b;
  a.s14 += b;
  a.s15 += b;
  return a;
}
occaFunction inline double16  operator -  (const double16 &a, const double16 &b) {
  return OCCA_DOUBLE16(a.x - b.x,
                       a.y - b.y,
                       a.z - b.z,
                       a.w - b.w,
                       a.s4 - b.s4,
                       a.s5 - b.s5,
                       a.s6 - b.s6,
                       a.s7 - b.s7,
                       a.s8 - b.s8,
                       a.s9 - b.s9,
                       a.s10 - b.s10,
                       a.s11 - b.s11,
                       a.s12 - b.s12,
                       a.s13 - b.s13,
                       a.s14 - b.s14,
                       a.s15 - b.s15);
}

occaFunction inline double16  operator -  (const double &a, const double16 &b) {
  return OCCA_DOUBLE16(a - b.x,
                       a - b.y,
                       a - b.z,
                       a - b.w,
                       a - b.s4,
                       a - b.s5,
                       a - b.s6,
                       a - b.s7,
                       a - b.s8,
                       a - b.s9,
                       a - b.s10,
                       a - b.s11,
                       a - b.s12,
                       a - b.s13,
                       a - b.s14,
                       a - b.s15);
}

occaFunction inline double16  operator -  (const double16 &a, const double &b) {
  return OCCA_DOUBLE16(a.x - b,
                       a.y - b,
                       a.z - b,
                       a.w - b,
                       a.s4 - b,
                       a.s5 - b,
                       a.s6 - b,
                       a.s7 - b,
                       a.s8 - b,
                       a.s9 - b,
                       a.s10 - b,
                       a.s11 - b,
                       a.s12 - b,
                       a.s13 - b,
                       a.s14 - b,
                       a.s15 - b);
}

occaFunction inline double16& operator -= (      double16 &a, const double16 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  a.s4 -= b.s4;
  a.s5 -= b.s5;
  a.s6 -= b.s6;
  a.s7 -= b.s7;
  a.s8 -= b.s8;
  a.s9 -= b.s9;
  a.s10 -= b.s10;
  a.s11 -= b.s11;
  a.s12 -= b.s12;
  a.s13 -= b.s13;
  a.s14 -= b.s14;
  a.s15 -= b.s15;
  return a;
}

occaFunction inline double16& operator -= (      double16 &a, const double &b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  a.s4 -= b;
  a.s5 -= b;
  a.s6 -= b;
  a.s7 -= b;
  a.s8 -= b;
  a.s9 -= b;
  a.s10 -= b;
  a.s11 -= b;
  a.s12 -= b;
  a.s13 -= b;
  a.s14 -= b;
  a.s15 -= b;
  return a;
}
occaFunction inline double16  operator *  (const double16 &a, const double16 &b) {
  return OCCA_DOUBLE16(a.x * b.x,
                       a.y * b.y,
                       a.z * b.z,
                       a.w * b.w,
                       a.s4 * b.s4,
                       a.s5 * b.s5,
                       a.s6 * b.s6,
                       a.s7 * b.s7,
                       a.s8 * b.s8,
                       a.s9 * b.s9,
                       a.s10 * b.s10,
                       a.s11 * b.s11,
                       a.s12 * b.s12,
                       a.s13 * b.s13,
                       a.s14 * b.s14,
                       a.s15 * b.s15);
}

occaFunction inline double16  operator *  (const double &a, const double16 &b) {
  return OCCA_DOUBLE16(a * b.x,
                       a * b.y,
                       a * b.z,
                       a * b.w,
                       a * b.s4,
                       a * b.s5,
                       a * b.s6,
                       a * b.s7,
                       a * b.s8,
                       a * b.s9,
                       a * b.s10,
                       a * b.s11,
                       a * b.s12,
                       a * b.s13,
                       a * b.s14,
                       a * b.s15);
}

occaFunction inline double16  operator *  (const double16 &a, const double &b) {
  return OCCA_DOUBLE16(a.x * b,
                       a.y * b,
                       a.z * b,
                       a.w * b,
                       a.s4 * b,
                       a.s5 * b,
                       a.s6 * b,
                       a.s7 * b,
                       a.s8 * b,
                       a.s9 * b,
                       a.s10 * b,
                       a.s11 * b,
                       a.s12 * b,
                       a.s13 * b,
                       a.s14 * b,
                       a.s15 * b);
}

occaFunction inline double16& operator *= (      double16 &a, const double16 &b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  a.s4 *= b.s4;
  a.s5 *= b.s5;
  a.s6 *= b.s6;
  a.s7 *= b.s7;
  a.s8 *= b.s8;
  a.s9 *= b.s9;
  a.s10 *= b.s10;
  a.s11 *= b.s11;
  a.s12 *= b.s12;
  a.s13 *= b.s13;
  a.s14 *= b.s14;
  a.s15 *= b.s15;
  return a;
}

occaFunction inline double16& operator *= (      double16 &a, const double &b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  a.s4 *= b;
  a.s5 *= b;
  a.s6 *= b;
  a.s7 *= b;
  a.s8 *= b;
  a.s9 *= b;
  a.s10 *= b;
  a.s11 *= b;
  a.s12 *= b;
  a.s13 *= b;
  a.s14 *= b;
  a.s15 *= b;
  return a;
}
occaFunction inline double16  operator /  (const double16 &a, const double16 &b) {
  return OCCA_DOUBLE16(a.x / b.x,
                       a.y / b.y,
                       a.z / b.z,
                       a.w / b.w,
                       a.s4 / b.s4,
                       a.s5 / b.s5,
                       a.s6 / b.s6,
                       a.s7 / b.s7,
                       a.s8 / b.s8,
                       a.s9 / b.s9,
                       a.s10 / b.s10,
                       a.s11 / b.s11,
                       a.s12 / b.s12,
                       a.s13 / b.s13,
                       a.s14 / b.s14,
                       a.s15 / b.s15);
}

occaFunction inline double16  operator /  (const double &a, const double16 &b) {
  return OCCA_DOUBLE16(a / b.x,
                       a / b.y,
                       a / b.z,
                       a / b.w,
                       a / b.s4,
                       a / b.s5,
                       a / b.s6,
                       a / b.s7,
                       a / b.s8,
                       a / b.s9,
                       a / b.s10,
                       a / b.s11,
                       a / b.s12,
                       a / b.s13,
                       a / b.s14,
                       a / b.s15);
}

occaFunction inline double16  operator /  (const double16 &a, const double &b) {
  return OCCA_DOUBLE16(a.x / b,
                       a.y / b,
                       a.z / b,
                       a.w / b,
                       a.s4 / b,
                       a.s5 / b,
                       a.s6 / b,
                       a.s7 / b,
                       a.s8 / b,
                       a.s9 / b,
                       a.s10 / b,
                       a.s11 / b,
                       a.s12 / b,
                       a.s13 / b,
                       a.s14 / b,
                       a.s15 / b);
}

occaFunction inline double16& operator /= (      double16 &a, const double16 &b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  a.s4 /= b.s4;
  a.s5 /= b.s5;
  a.s6 /= b.s6;
  a.s7 /= b.s7;
  a.s8 /= b.s8;
  a.s9 /= b.s9;
  a.s10 /= b.s10;
  a.s11 /= b.s11;
  a.s12 /= b.s12;
  a.s13 /= b.s13;
  a.s14 /= b.s14;
  a.s15 /= b.s15;
  return a;
}

occaFunction inline double16& operator /= (      double16 &a, const double &b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  a.s4 /= b;
  a.s5 /= b;
  a.s6 /= b;
  a.s7 /= b;
  a.s8 /= b;
  a.s9 /= b;
  a.s10 /= b;
  a.s11 /= b;
  a.s12 /= b;
  a.s13 /= b;
  a.s14 /= b;
  a.s15 /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const double16& a) {
  out << "[" << a.x << ", "
             << a.y << ", "
             << a.z << ", "
             << a.w << ", "
             << a.s4 << ", "
             << a.s5 << ", "
             << a.s6 << ", "
             << a.s7 << ", "
             << a.s8 << ", "
             << a.s9 << ", "
             << a.s10 << ", "
             << a.s11 << ", "
             << a.s12 << ", "
             << a.s13 << ", "
             << a.s14 << ", "
             << a.s15
      << "]\n";

  return out;
}
#endif

//======================================


#  ifndef OCCA_IN_KERNEL
}
#  endif

#endif
#if OCCA_USING_OPENCL
#  define OCCA_BOOL2(a, b) (bool2)(a, b)
#  define OCCA_BOOL4(a, b, c, d) (bool4)(a, b, c, d)
#  define OCCA_BOOL3(a, b, c) (bool3)(a, b, c)
#  define OCCA_BOOL8(a, b, c, d, e, f, g, h) (bool8)(a, b, c, d, e, f, g, h)
#  define OCCA_BOOL16(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) (bool16)(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
#  define OCCA_CHAR2(a, b) (char2)(a, b)
#  define OCCA_CHAR4(a, b, c, d) (char4)(a, b, c, d)
#  define OCCA_CHAR3(a, b, c) (char3)(a, b, c)
#  define OCCA_CHAR8(a, b, c, d, e, f, g, h) (char8)(a, b, c, d, e, f, g, h)
#  define OCCA_CHAR16(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) (char16)(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
#  define OCCA_SHORT2(a, b) (short2)(a, b)
#  define OCCA_SHORT4(a, b, c, d) (short4)(a, b, c, d)
#  define OCCA_SHORT3(a, b, c) (short3)(a, b, c)
#  define OCCA_SHORT8(a, b, c, d, e, f, g, h) (short8)(a, b, c, d, e, f, g, h)
#  define OCCA_SHORT16(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) (short16)(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
#  define OCCA_INT2(a, b) (int2)(a, b)
#  define OCCA_INT4(a, b, c, d) (int4)(a, b, c, d)
#  define OCCA_INT3(a, b, c) (int3)(a, b, c)
#  define OCCA_INT8(a, b, c, d, e, f, g, h) (int8)(a, b, c, d, e, f, g, h)
#  define OCCA_INT16(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) (int16)(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
#  define OCCA_LONG2(a, b) (long2)(a, b)
#  define OCCA_LONG4(a, b, c, d) (long4)(a, b, c, d)
#  define OCCA_LONG3(a, b, c) (long3)(a, b, c)
#  define OCCA_LONG8(a, b, c, d, e, f, g, h) (long8)(a, b, c, d, e, f, g, h)
#  define OCCA_LONG16(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) (long16)(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
#  define OCCA_FLOAT2(a, b) (float2)(a, b)
#  define OCCA_FLOAT4(a, b, c, d) (float4)(a, b, c, d)
#  define OCCA_FLOAT3(a, b, c) (float3)(a, b, c)
#  define OCCA_FLOAT8(a, b, c, d, e, f, g, h) (float8)(a, b, c, d, e, f, g, h)
#  define OCCA_FLOAT16(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) (float16)(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
#  define OCCA_DOUBLE2(a, b) (double2)(a, b)
#  define OCCA_DOUBLE4(a, b, c, d) (double4)(a, b, c, d)
#  define OCCA_DOUBLE3(a, b, c) (double3)(a, b, c)
#  define OCCA_DOUBLE8(a, b, c, d, e, f, g, h) (double8)(a, b, c, d, e, f, g, h)
#  define OCCA_DOUBLE16(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) (double16)(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
#endif


#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS || OCCA_USING_CUDA))
#  if !defined(OCCA_IN_KERNEL)
namespace occa {
#  endif

occaFunction inline char length(const char2 &v) {
  return sqrt(v.x*v.x+v.y*v.y);
}

occaFunction inline char length(const char4 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w);
}

occaFunction inline char length(const char8 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w+v.s4*v.s4+v.s5*v.s5+v.s6*v.s6+v.s7*v.s7);
}

occaFunction inline char length(const char16 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w+v.s4*v.s4+v.s5*v.s5+v.s6*v.s6+v.s7*v.s7+v.s8*v.s8+v.s9*v.s9+v.s10*v.s10+v.s11*v.s11+v.s12*v.s12+v.s13*v.s13+v.s14*v.s14+v.s15*v.s15);
}

occaFunction inline short length(const short2 &v) {
  return sqrt(v.x*v.x+v.y*v.y);
}

occaFunction inline short length(const short4 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w);
}

occaFunction inline short length(const short8 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w+v.s4*v.s4+v.s5*v.s5+v.s6*v.s6+v.s7*v.s7);
}

occaFunction inline short length(const short16 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w+v.s4*v.s4+v.s5*v.s5+v.s6*v.s6+v.s7*v.s7+v.s8*v.s8+v.s9*v.s9+v.s10*v.s10+v.s11*v.s11+v.s12*v.s12+v.s13*v.s13+v.s14*v.s14+v.s15*v.s15);
}

occaFunction inline int length(const int2 &v) {
  return sqrt(v.x*v.x+v.y*v.y);
}

occaFunction inline int length(const int4 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w);
}

occaFunction inline int length(const int8 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w+v.s4*v.s4+v.s5*v.s5+v.s6*v.s6+v.s7*v.s7);
}

occaFunction inline int length(const int16 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w+v.s4*v.s4+v.s5*v.s5+v.s6*v.s6+v.s7*v.s7+v.s8*v.s8+v.s9*v.s9+v.s10*v.s10+v.s11*v.s11+v.s12*v.s12+v.s13*v.s13+v.s14*v.s14+v.s15*v.s15);
}

occaFunction inline long length(const long2 &v) {
  return sqrt(v.x*v.x+v.y*v.y);
}

occaFunction inline long length(const long4 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w);
}

occaFunction inline long length(const long8 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w+v.s4*v.s4+v.s5*v.s5+v.s6*v.s6+v.s7*v.s7);
}

occaFunction inline long length(const long16 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w+v.s4*v.s4+v.s5*v.s5+v.s6*v.s6+v.s7*v.s7+v.s8*v.s8+v.s9*v.s9+v.s10*v.s10+v.s11*v.s11+v.s12*v.s12+v.s13*v.s13+v.s14*v.s14+v.s15*v.s15);
}

occaFunction inline float length(const float2 &v) {
  return sqrt(v.x*v.x+v.y*v.y);
}

occaFunction inline float length(const float4 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w);
}

occaFunction inline float length(const float8 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w+v.s4*v.s4+v.s5*v.s5+v.s6*v.s6+v.s7*v.s7);
}

occaFunction inline float length(const float16 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w+v.s4*v.s4+v.s5*v.s5+v.s6*v.s6+v.s7*v.s7+v.s8*v.s8+v.s9*v.s9+v.s10*v.s10+v.s11*v.s11+v.s12*v.s12+v.s13*v.s13+v.s14*v.s14+v.s15*v.s15);
}

occaFunction inline double length(const double2 &v) {
  return sqrt(v.x*v.x+v.y*v.y);
}

occaFunction inline double length(const double4 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w);
}

occaFunction inline double length(const double8 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w+v.s4*v.s4+v.s5*v.s5+v.s6*v.s6+v.s7*v.s7);
}

occaFunction inline double length(const double16 &v) {
  return sqrt(v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w+v.s4*v.s4+v.s5*v.s5+v.s6*v.s6+v.s7*v.s7+v.s8*v.s8+v.s9*v.s9+v.s10*v.s10+v.s11*v.s11+v.s12*v.s12+v.s13*v.s13+v.s14*v.s14+v.s15*v.s15);
}

occaFunction inline char2 normalize(const char2 &v) {
  const char invNorm = (1.0 / length(v));
  return OCCA_CHAR2(invNorm*v.x,invNorm*v.y);
}

occaFunction inline char4 normalize(const char4 &v) {
  const char invNorm = (1.0 / length(v));
  return OCCA_CHAR4(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w);
}

occaFunction inline char8 normalize(const char8 &v) {
  const char invNorm = (1.0 / length(v));
  return OCCA_CHAR8(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w,invNorm*v.s4,invNorm*v.s5,invNorm*v.s6,invNorm*v.s7);
}

occaFunction inline char16 normalize(const char16 &v) {
  const char invNorm = (1.0 / length(v));
  return OCCA_CHAR16(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w,invNorm*v.s4,invNorm*v.s5,invNorm*v.s6,invNorm*v.s7,invNorm*v.s8,invNorm*v.s9,invNorm*v.s10,invNorm*v.s11,invNorm*v.s12,invNorm*v.s13,invNorm*v.s14,invNorm*v.s15);
}

occaFunction inline short2 normalize(const short2 &v) {
  const short invNorm = (1.0 / length(v));
  return OCCA_SHORT2(invNorm*v.x,invNorm*v.y);
}

occaFunction inline short4 normalize(const short4 &v) {
  const short invNorm = (1.0 / length(v));
  return OCCA_SHORT4(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w);
}

occaFunction inline short8 normalize(const short8 &v) {
  const short invNorm = (1.0 / length(v));
  return OCCA_SHORT8(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w,invNorm*v.s4,invNorm*v.s5,invNorm*v.s6,invNorm*v.s7);
}

occaFunction inline short16 normalize(const short16 &v) {
  const short invNorm = (1.0 / length(v));
  return OCCA_SHORT16(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w,invNorm*v.s4,invNorm*v.s5,invNorm*v.s6,invNorm*v.s7,invNorm*v.s8,invNorm*v.s9,invNorm*v.s10,invNorm*v.s11,invNorm*v.s12,invNorm*v.s13,invNorm*v.s14,invNorm*v.s15);
}

occaFunction inline int2 normalize(const int2 &v) {
  const int invNorm = (1.0 / length(v));
  return OCCA_INT2(invNorm*v.x,invNorm*v.y);
}

occaFunction inline int4 normalize(const int4 &v) {
  const int invNorm = (1.0 / length(v));
  return OCCA_INT4(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w);
}

occaFunction inline int8 normalize(const int8 &v) {
  const int invNorm = (1.0 / length(v));
  return OCCA_INT8(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w,invNorm*v.s4,invNorm*v.s5,invNorm*v.s6,invNorm*v.s7);
}

occaFunction inline int16 normalize(const int16 &v) {
  const int invNorm = (1.0 / length(v));
  return OCCA_INT16(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w,invNorm*v.s4,invNorm*v.s5,invNorm*v.s6,invNorm*v.s7,invNorm*v.s8,invNorm*v.s9,invNorm*v.s10,invNorm*v.s11,invNorm*v.s12,invNorm*v.s13,invNorm*v.s14,invNorm*v.s15);
}

occaFunction inline long2 normalize(const long2 &v) {
  const long invNorm = (1.0 / length(v));
  return OCCA_LONG2(invNorm*v.x,invNorm*v.y);
}

occaFunction inline long4 normalize(const long4 &v) {
  const long invNorm = (1.0 / length(v));
  return OCCA_LONG4(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w);
}

occaFunction inline long8 normalize(const long8 &v) {
  const long invNorm = (1.0 / length(v));
  return OCCA_LONG8(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w,invNorm*v.s4,invNorm*v.s5,invNorm*v.s6,invNorm*v.s7);
}

occaFunction inline long16 normalize(const long16 &v) {
  const long invNorm = (1.0 / length(v));
  return OCCA_LONG16(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w,invNorm*v.s4,invNorm*v.s5,invNorm*v.s6,invNorm*v.s7,invNorm*v.s8,invNorm*v.s9,invNorm*v.s10,invNorm*v.s11,invNorm*v.s12,invNorm*v.s13,invNorm*v.s14,invNorm*v.s15);
}

occaFunction inline float2 normalize(const float2 &v) {
  const float invNorm = (1.0 / length(v));
  return OCCA_FLOAT2(invNorm*v.x,invNorm*v.y);
}

occaFunction inline float4 normalize(const float4 &v) {
  const float invNorm = (1.0 / length(v));
  return OCCA_FLOAT4(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w);
}

occaFunction inline float8 normalize(const float8 &v) {
  const float invNorm = (1.0 / length(v));
  return OCCA_FLOAT8(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w,invNorm*v.s4,invNorm*v.s5,invNorm*v.s6,invNorm*v.s7);
}

occaFunction inline float16 normalize(const float16 &v) {
  const float invNorm = (1.0 / length(v));
  return OCCA_FLOAT16(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w,invNorm*v.s4,invNorm*v.s5,invNorm*v.s6,invNorm*v.s7,invNorm*v.s8,invNorm*v.s9,invNorm*v.s10,invNorm*v.s11,invNorm*v.s12,invNorm*v.s13,invNorm*v.s14,invNorm*v.s15);
}

occaFunction inline double2 normalize(const double2 &v) {
  const double invNorm = (1.0 / length(v));
  return OCCA_DOUBLE2(invNorm*v.x,invNorm*v.y);
}

occaFunction inline double4 normalize(const double4 &v) {
  const double invNorm = (1.0 / length(v));
  return OCCA_DOUBLE4(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w);
}

occaFunction inline double8 normalize(const double8 &v) {
  const double invNorm = (1.0 / length(v));
  return OCCA_DOUBLE8(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w,invNorm*v.s4,invNorm*v.s5,invNorm*v.s6,invNorm*v.s7);
}

occaFunction inline double16 normalize(const double16 &v) {
  const double invNorm = (1.0 / length(v));
  return OCCA_DOUBLE16(invNorm*v.x,invNorm*v.y,invNorm*v.z,invNorm*v.w,invNorm*v.s4,invNorm*v.s5,invNorm*v.s6,invNorm*v.s7,invNorm*v.s8,invNorm*v.s9,invNorm*v.s10,invNorm*v.s11,invNorm*v.s12,invNorm*v.s13,invNorm*v.s14,invNorm*v.s15);
}

occaFunction inline char dot(const char2 &a, const char2 &b) {
  return (a.x*b.x+a.y*b.y);
}

occaFunction inline char dot(const char4 &a, const char4 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w);
}

occaFunction inline char dot(const char8 &a, const char8 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w+a.s4*b.s4+a.s5*b.s5+a.s6*b.s6+a.s7*b.s7);
}

occaFunction inline char dot(const char16 &a, const char16 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w+a.s4*b.s4+a.s5*b.s5+a.s6*b.s6+a.s7*b.s7+a.s8*b.s8+a.s9*b.s9+a.s10*b.s10+a.s11*b.s11+a.s12*b.s12+a.s13*b.s13+a.s14*b.s14+a.s15*b.s15);
}

occaFunction inline short dot(const short2 &a, const short2 &b) {
  return (a.x*b.x+a.y*b.y);
}

occaFunction inline short dot(const short4 &a, const short4 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w);
}

occaFunction inline short dot(const short8 &a, const short8 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w+a.s4*b.s4+a.s5*b.s5+a.s6*b.s6+a.s7*b.s7);
}

occaFunction inline short dot(const short16 &a, const short16 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w+a.s4*b.s4+a.s5*b.s5+a.s6*b.s6+a.s7*b.s7+a.s8*b.s8+a.s9*b.s9+a.s10*b.s10+a.s11*b.s11+a.s12*b.s12+a.s13*b.s13+a.s14*b.s14+a.s15*b.s15);
}

occaFunction inline int dot(const int2 &a, const int2 &b) {
  return (a.x*b.x+a.y*b.y);
}

occaFunction inline int dot(const int4 &a, const int4 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w);
}

occaFunction inline int dot(const int8 &a, const int8 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w+a.s4*b.s4+a.s5*b.s5+a.s6*b.s6+a.s7*b.s7);
}

occaFunction inline int dot(const int16 &a, const int16 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w+a.s4*b.s4+a.s5*b.s5+a.s6*b.s6+a.s7*b.s7+a.s8*b.s8+a.s9*b.s9+a.s10*b.s10+a.s11*b.s11+a.s12*b.s12+a.s13*b.s13+a.s14*b.s14+a.s15*b.s15);
}

occaFunction inline long dot(const long2 &a, const long2 &b) {
  return (a.x*b.x+a.y*b.y);
}

occaFunction inline long dot(const long4 &a, const long4 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w);
}

occaFunction inline long dot(const long8 &a, const long8 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w+a.s4*b.s4+a.s5*b.s5+a.s6*b.s6+a.s7*b.s7);
}

occaFunction inline long dot(const long16 &a, const long16 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w+a.s4*b.s4+a.s5*b.s5+a.s6*b.s6+a.s7*b.s7+a.s8*b.s8+a.s9*b.s9+a.s10*b.s10+a.s11*b.s11+a.s12*b.s12+a.s13*b.s13+a.s14*b.s14+a.s15*b.s15);
}

occaFunction inline float dot(const float2 &a, const float2 &b) {
  return (a.x*b.x+a.y*b.y);
}

occaFunction inline float dot(const float4 &a, const float4 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w);
}

occaFunction inline float dot(const float8 &a, const float8 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w+a.s4*b.s4+a.s5*b.s5+a.s6*b.s6+a.s7*b.s7);
}

occaFunction inline float dot(const float16 &a, const float16 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w+a.s4*b.s4+a.s5*b.s5+a.s6*b.s6+a.s7*b.s7+a.s8*b.s8+a.s9*b.s9+a.s10*b.s10+a.s11*b.s11+a.s12*b.s12+a.s13*b.s13+a.s14*b.s14+a.s15*b.s15);
}

occaFunction inline double dot(const double2 &a, const double2 &b) {
  return (a.x*b.x+a.y*b.y);
}

occaFunction inline double dot(const double4 &a, const double4 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w);
}

occaFunction inline double dot(const double8 &a, const double8 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w+a.s4*b.s4+a.s5*b.s5+a.s6*b.s6+a.s7*b.s7);
}

occaFunction inline double dot(const double16 &a, const double16 &b) {
  return (a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w+a.s4*b.s4+a.s5*b.s5+a.s6*b.s6+a.s7*b.s7+a.s8*b.s8+a.s9*b.s9+a.s10*b.s10+a.s11*b.s11+a.s12*b.s12+a.s13*b.s13+a.s14*b.s14+a.s15*b.s15);
}

occaFunction inline char clamp(const char val, const char min, const char max) {
  return (val < min) ? min : ((max < val) ? max : val);
}

occaFunction inline char2 clamp(const char2 &v, const char min, const char max) {
  return OCCA_CHAR2(clamp(v.x,min,max),clamp(v.y,min,max));
}

occaFunction inline char4 clamp(const char4 &v, const char min, const char max) {
  return OCCA_CHAR4(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max));
}

occaFunction inline char8 clamp(const char8 &v, const char min, const char max) {
  return OCCA_CHAR8(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max),clamp(v.s4,min,max),clamp(v.s5,min,max),clamp(v.s6,min,max),clamp(v.s7,min,max));
}

occaFunction inline char16 clamp(const char16 &v, const char min, const char max) {
  return OCCA_CHAR16(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max),clamp(v.s4,min,max),clamp(v.s5,min,max),clamp(v.s6,min,max),clamp(v.s7,min,max),clamp(v.s8,min,max),clamp(v.s9,min,max),clamp(v.s10,min,max),clamp(v.s11,min,max),clamp(v.s12,min,max),clamp(v.s13,min,max),clamp(v.s14,min,max),clamp(v.s15,min,max));
}

occaFunction inline short clamp(const short val, const short min, const short max) {
  return (val < min) ? min : ((max < val) ? max : val);
}

occaFunction inline short2 clamp(const short2 &v, const short min, const short max) {
  return OCCA_SHORT2(clamp(v.x,min,max),clamp(v.y,min,max));
}

occaFunction inline short4 clamp(const short4 &v, const short min, const short max) {
  return OCCA_SHORT4(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max));
}

occaFunction inline short8 clamp(const short8 &v, const short min, const short max) {
  return OCCA_SHORT8(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max),clamp(v.s4,min,max),clamp(v.s5,min,max),clamp(v.s6,min,max),clamp(v.s7,min,max));
}

occaFunction inline short16 clamp(const short16 &v, const short min, const short max) {
  return OCCA_SHORT16(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max),clamp(v.s4,min,max),clamp(v.s5,min,max),clamp(v.s6,min,max),clamp(v.s7,min,max),clamp(v.s8,min,max),clamp(v.s9,min,max),clamp(v.s10,min,max),clamp(v.s11,min,max),clamp(v.s12,min,max),clamp(v.s13,min,max),clamp(v.s14,min,max),clamp(v.s15,min,max));
}

occaFunction inline int clamp(const int val, const int min, const int max) {
  return (val < min) ? min : ((max < val) ? max : val);
}

occaFunction inline int2 clamp(const int2 &v, const int min, const int max) {
  return OCCA_INT2(clamp(v.x,min,max),clamp(v.y,min,max));
}

occaFunction inline int4 clamp(const int4 &v, const int min, const int max) {
  return OCCA_INT4(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max));
}

occaFunction inline int8 clamp(const int8 &v, const int min, const int max) {
  return OCCA_INT8(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max),clamp(v.s4,min,max),clamp(v.s5,min,max),clamp(v.s6,min,max),clamp(v.s7,min,max));
}

occaFunction inline int16 clamp(const int16 &v, const int min, const int max) {
  return OCCA_INT16(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max),clamp(v.s4,min,max),clamp(v.s5,min,max),clamp(v.s6,min,max),clamp(v.s7,min,max),clamp(v.s8,min,max),clamp(v.s9,min,max),clamp(v.s10,min,max),clamp(v.s11,min,max),clamp(v.s12,min,max),clamp(v.s13,min,max),clamp(v.s14,min,max),clamp(v.s15,min,max));
}

occaFunction inline long clamp(const long val, const long min, const long max) {
  return (val < min) ? min : ((max < val) ? max : val);
}

occaFunction inline long2 clamp(const long2 &v, const long min, const long max) {
  return OCCA_LONG2(clamp(v.x,min,max),clamp(v.y,min,max));
}

occaFunction inline long4 clamp(const long4 &v, const long min, const long max) {
  return OCCA_LONG4(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max));
}

occaFunction inline long8 clamp(const long8 &v, const long min, const long max) {
  return OCCA_LONG8(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max),clamp(v.s4,min,max),clamp(v.s5,min,max),clamp(v.s6,min,max),clamp(v.s7,min,max));
}

occaFunction inline long16 clamp(const long16 &v, const long min, const long max) {
  return OCCA_LONG16(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max),clamp(v.s4,min,max),clamp(v.s5,min,max),clamp(v.s6,min,max),clamp(v.s7,min,max),clamp(v.s8,min,max),clamp(v.s9,min,max),clamp(v.s10,min,max),clamp(v.s11,min,max),clamp(v.s12,min,max),clamp(v.s13,min,max),clamp(v.s14,min,max),clamp(v.s15,min,max));
}

occaFunction inline float clamp(const float val, const float min, const float max) {
  return (val < min) ? min : ((max < val) ? max : val);
}

occaFunction inline float2 clamp(const float2 &v, const float min, const float max) {
  return OCCA_FLOAT2(clamp(v.x,min,max),clamp(v.y,min,max));
}

occaFunction inline float4 clamp(const float4 &v, const float min, const float max) {
  return OCCA_FLOAT4(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max));
}

occaFunction inline float8 clamp(const float8 &v, const float min, const float max) {
  return OCCA_FLOAT8(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max),clamp(v.s4,min,max),clamp(v.s5,min,max),clamp(v.s6,min,max),clamp(v.s7,min,max));
}

occaFunction inline float16 clamp(const float16 &v, const float min, const float max) {
  return OCCA_FLOAT16(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max),clamp(v.s4,min,max),clamp(v.s5,min,max),clamp(v.s6,min,max),clamp(v.s7,min,max),clamp(v.s8,min,max),clamp(v.s9,min,max),clamp(v.s10,min,max),clamp(v.s11,min,max),clamp(v.s12,min,max),clamp(v.s13,min,max),clamp(v.s14,min,max),clamp(v.s15,min,max));
}

occaFunction inline double clamp(const double val, const double min, const double max) {
  return (val < min) ? min : ((max < val) ? max : val);
}

occaFunction inline double2 clamp(const double2 &v, const double min, const double max) {
  return OCCA_DOUBLE2(clamp(v.x,min,max),clamp(v.y,min,max));
}

occaFunction inline double4 clamp(const double4 &v, const double min, const double max) {
  return OCCA_DOUBLE4(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max));
}

occaFunction inline double8 clamp(const double8 &v, const double min, const double max) {
  return OCCA_DOUBLE8(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max),clamp(v.s4,min,max),clamp(v.s5,min,max),clamp(v.s6,min,max),clamp(v.s7,min,max));
}

occaFunction inline double16 clamp(const double16 &v, const double min, const double max) {
  return OCCA_DOUBLE16(clamp(v.x,min,max),clamp(v.y,min,max),clamp(v.z,min,max),clamp(v.w,min,max),clamp(v.s4,min,max),clamp(v.s5,min,max),clamp(v.s6,min,max),clamp(v.s7,min,max),clamp(v.s8,min,max),clamp(v.s9,min,max),clamp(v.s10,min,max),clamp(v.s11,min,max),clamp(v.s12,min,max),clamp(v.s13,min,max),clamp(v.s14,min,max),clamp(v.s15,min,max));
}

occaFunction inline float3 cross(const float3 &a, const float3 &b) {
  return OCCA_FLOAT3(a.z*b.y - b.z*a.y,
                     a.x*b.z - b.x*a.z,
                     a.y*b.x - b.y*a.x);
}

occaFunction inline double3 cross(const double3 &a, const double3 &b) {
  return OCCA_DOUBLE3(a.z*b.y - b.z*a.y,
                     a.x*b.z - b.x*a.z,
                     a.y*b.x - b.y*a.x);
}

#  ifndef OCCA_IN_KERNEL
}
#  endif
#endif
#endif
