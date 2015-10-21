#if (!defined(OCCA_IN_KERNEL) || (!OCCA_USING_OPENCL))
#  if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
#    include <iostream>
#  endif

#  ifndef OCCA_IN_KERNEL
#    define occaFunction
namespace occa {
#  endif

//---[ bool2 ]--------------------------
#  define OCCA_BOOL2_CONSTRUCTOR bool2
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

occaFunction inline bool2  operator +  (const bool2 &a, const bool2 &b){
  return OCCA_BOOL2_CONSTRUCTOR(a.x + b.x,
                                a.y + b.y);
}

occaFunction inline bool2  operator +  (const bool &a, const bool2 &b){
  return OCCA_BOOL2_CONSTRUCTOR(a + b.x,
                                a + b.y);
}

occaFunction inline bool2  operator +  (const bool2 &a, const bool &b){
  return OCCA_BOOL2_CONSTRUCTOR(a.x + b,
                                a.y + b);
}

occaFunction inline bool2& operator += (      bool2 &a, const bool2 &b){
  a.x += b.x;
  a.y += b.y;
  return a;
}

occaFunction inline bool2& operator += (      bool2 &a, const bool &b){
  a.x += b;
  a.y += b;
  return a;
}
occaFunction inline bool2  operator -  (const bool2 &a, const bool2 &b){
  return OCCA_BOOL2_CONSTRUCTOR(a.x - b.x,
                                a.y - b.y);
}

occaFunction inline bool2  operator -  (const bool &a, const bool2 &b){
  return OCCA_BOOL2_CONSTRUCTOR(a - b.x,
                                a - b.y);
}

occaFunction inline bool2  operator -  (const bool2 &a, const bool &b){
  return OCCA_BOOL2_CONSTRUCTOR(a.x - b,
                                a.y - b);
}

occaFunction inline bool2& operator -= (      bool2 &a, const bool2 &b){
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

occaFunction inline bool2& operator -= (      bool2 &a, const bool &b){
  a.x -= b;
  a.y -= b;
  return a;
}
occaFunction inline bool2  operator *  (const bool2 &a, const bool2 &b){
  return OCCA_BOOL2_CONSTRUCTOR(a.x * b.x,
                                a.y * b.y);
}

occaFunction inline bool2  operator *  (const bool &a, const bool2 &b){
  return OCCA_BOOL2_CONSTRUCTOR(a * b.x,
                                a * b.y);
}

occaFunction inline bool2  operator *  (const bool2 &a, const bool &b){
  return OCCA_BOOL2_CONSTRUCTOR(a.x * b,
                                a.y * b);
}

occaFunction inline bool2& operator *= (      bool2 &a, const bool2 &b){
  a.x *= b.x;
  a.y *= b.y;
  return a;
}

occaFunction inline bool2& operator *= (      bool2 &a, const bool &b){
  a.x *= b;
  a.y *= b;
  return a;
}
occaFunction inline bool2  operator /  (const bool2 &a, const bool2 &b){
  return OCCA_BOOL2_CONSTRUCTOR(a.x / b.x,
                                a.y / b.y);
}

occaFunction inline bool2  operator /  (const bool &a, const bool2 &b){
  return OCCA_BOOL2_CONSTRUCTOR(a / b.x,
                                a / b.y);
}

occaFunction inline bool2  operator /  (const bool2 &a, const bool &b){
  return OCCA_BOOL2_CONSTRUCTOR(a.x / b,
                                a.y / b);
}

occaFunction inline bool2& operator /= (      bool2 &a, const bool2 &b){
  a.x /= b.x;
  a.y /= b.y;
  return a;
}

occaFunction inline bool2& operator /= (      bool2 &a, const bool &b){
  a.x /= b;
  a.y /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const bool2& a){
  out << "[" << (a.x ? "true" : "false") << ", "
             << (a.y ? "true" : "false")
      << "]\n";

  return out;
}
#endif

//======================================


//---[ bool4 ]--------------------------
#  define OCCA_BOOL4_CONSTRUCTOR bool4
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

occaFunction inline bool4  operator +  (const bool4 &a, const bool4 &b){
  return OCCA_BOOL4_CONSTRUCTOR(a.x + b.x,
                                a.y + b.y,
                                a.z + b.z,
                                a.w + b.w);
}

occaFunction inline bool4  operator +  (const bool &a, const bool4 &b){
  return OCCA_BOOL4_CONSTRUCTOR(a + b.x,
                                a + b.y,
                                a + b.z,
                                a + b.w);
}

occaFunction inline bool4  operator +  (const bool4 &a, const bool &b){
  return OCCA_BOOL4_CONSTRUCTOR(a.x + b,
                                a.y + b,
                                a.z + b,
                                a.w + b);
}

occaFunction inline bool4& operator += (      bool4 &a, const bool4 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

occaFunction inline bool4& operator += (      bool4 &a, const bool &b){
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  return a;
}
occaFunction inline bool4  operator -  (const bool4 &a, const bool4 &b){
  return OCCA_BOOL4_CONSTRUCTOR(a.x - b.x,
                                a.y - b.y,
                                a.z - b.z,
                                a.w - b.w);
}

occaFunction inline bool4  operator -  (const bool &a, const bool4 &b){
  return OCCA_BOOL4_CONSTRUCTOR(a - b.x,
                                a - b.y,
                                a - b.z,
                                a - b.w);
}

occaFunction inline bool4  operator -  (const bool4 &a, const bool &b){
  return OCCA_BOOL4_CONSTRUCTOR(a.x - b,
                                a.y - b,
                                a.z - b,
                                a.w - b);
}

occaFunction inline bool4& operator -= (      bool4 &a, const bool4 &b){
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  return a;
}

occaFunction inline bool4& operator -= (      bool4 &a, const bool &b){
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  return a;
}
occaFunction inline bool4  operator *  (const bool4 &a, const bool4 &b){
  return OCCA_BOOL4_CONSTRUCTOR(a.x * b.x,
                                a.y * b.y,
                                a.z * b.z,
                                a.w * b.w);
}

occaFunction inline bool4  operator *  (const bool &a, const bool4 &b){
  return OCCA_BOOL4_CONSTRUCTOR(a * b.x,
                                a * b.y,
                                a * b.z,
                                a * b.w);
}

occaFunction inline bool4  operator *  (const bool4 &a, const bool &b){
  return OCCA_BOOL4_CONSTRUCTOR(a.x * b,
                                a.y * b,
                                a.z * b,
                                a.w * b);
}

occaFunction inline bool4& operator *= (      bool4 &a, const bool4 &b){
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  return a;
}

occaFunction inline bool4& operator *= (      bool4 &a, const bool &b){
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}
occaFunction inline bool4  operator /  (const bool4 &a, const bool4 &b){
  return OCCA_BOOL4_CONSTRUCTOR(a.x / b.x,
                                a.y / b.y,
                                a.z / b.z,
                                a.w / b.w);
}

occaFunction inline bool4  operator /  (const bool &a, const bool4 &b){
  return OCCA_BOOL4_CONSTRUCTOR(a / b.x,
                                a / b.y,
                                a / b.z,
                                a / b.w);
}

occaFunction inline bool4  operator /  (const bool4 &a, const bool &b){
  return OCCA_BOOL4_CONSTRUCTOR(a.x / b,
                                a.y / b,
                                a.z / b,
                                a.w / b);
}

occaFunction inline bool4& operator /= (      bool4 &a, const bool4 &b){
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  return a;
}

occaFunction inline bool4& operator /= (      bool4 &a, const bool &b){
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const bool4& a){
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
typedef bool4 bool3;
//======================================


//---[ bool8 ]--------------------------
#  define OCCA_BOOL8_CONSTRUCTOR bool8
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

occaFunction inline bool8  operator +  (const bool8 &a, const bool8 &b){
  return OCCA_BOOL8_CONSTRUCTOR(a.x + b.x,
                                a.y + b.y,
                                a.z + b.z,
                                a.w + b.w,
                                a.s4 + b.s4,
                                a.s5 + b.s5,
                                a.s6 + b.s6,
                                a.s7 + b.s7);
}

occaFunction inline bool8  operator +  (const bool &a, const bool8 &b){
  return OCCA_BOOL8_CONSTRUCTOR(a + b.x,
                                a + b.y,
                                a + b.z,
                                a + b.w,
                                a + b.s4,
                                a + b.s5,
                                a + b.s6,
                                a + b.s7);
}

occaFunction inline bool8  operator +  (const bool8 &a, const bool &b){
  return OCCA_BOOL8_CONSTRUCTOR(a.x + b,
                                a.y + b,
                                a.z + b,
                                a.w + b,
                                a.s4 + b,
                                a.s5 + b,
                                a.s6 + b,
                                a.s7 + b);
}

occaFunction inline bool8& operator += (      bool8 &a, const bool8 &b){
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

occaFunction inline bool8& operator += (      bool8 &a, const bool &b){
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
occaFunction inline bool8  operator -  (const bool8 &a, const bool8 &b){
  return OCCA_BOOL8_CONSTRUCTOR(a.x - b.x,
                                a.y - b.y,
                                a.z - b.z,
                                a.w - b.w,
                                a.s4 - b.s4,
                                a.s5 - b.s5,
                                a.s6 - b.s6,
                                a.s7 - b.s7);
}

occaFunction inline bool8  operator -  (const bool &a, const bool8 &b){
  return OCCA_BOOL8_CONSTRUCTOR(a - b.x,
                                a - b.y,
                                a - b.z,
                                a - b.w,
                                a - b.s4,
                                a - b.s5,
                                a - b.s6,
                                a - b.s7);
}

occaFunction inline bool8  operator -  (const bool8 &a, const bool &b){
  return OCCA_BOOL8_CONSTRUCTOR(a.x - b,
                                a.y - b,
                                a.z - b,
                                a.w - b,
                                a.s4 - b,
                                a.s5 - b,
                                a.s6 - b,
                                a.s7 - b);
}

occaFunction inline bool8& operator -= (      bool8 &a, const bool8 &b){
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

occaFunction inline bool8& operator -= (      bool8 &a, const bool &b){
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
occaFunction inline bool8  operator *  (const bool8 &a, const bool8 &b){
  return OCCA_BOOL8_CONSTRUCTOR(a.x * b.x,
                                a.y * b.y,
                                a.z * b.z,
                                a.w * b.w,
                                a.s4 * b.s4,
                                a.s5 * b.s5,
                                a.s6 * b.s6,
                                a.s7 * b.s7);
}

occaFunction inline bool8  operator *  (const bool &a, const bool8 &b){
  return OCCA_BOOL8_CONSTRUCTOR(a * b.x,
                                a * b.y,
                                a * b.z,
                                a * b.w,
                                a * b.s4,
                                a * b.s5,
                                a * b.s6,
                                a * b.s7);
}

occaFunction inline bool8  operator *  (const bool8 &a, const bool &b){
  return OCCA_BOOL8_CONSTRUCTOR(a.x * b,
                                a.y * b,
                                a.z * b,
                                a.w * b,
                                a.s4 * b,
                                a.s5 * b,
                                a.s6 * b,
                                a.s7 * b);
}

occaFunction inline bool8& operator *= (      bool8 &a, const bool8 &b){
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

occaFunction inline bool8& operator *= (      bool8 &a, const bool &b){
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
occaFunction inline bool8  operator /  (const bool8 &a, const bool8 &b){
  return OCCA_BOOL8_CONSTRUCTOR(a.x / b.x,
                                a.y / b.y,
                                a.z / b.z,
                                a.w / b.w,
                                a.s4 / b.s4,
                                a.s5 / b.s5,
                                a.s6 / b.s6,
                                a.s7 / b.s7);
}

occaFunction inline bool8  operator /  (const bool &a, const bool8 &b){
  return OCCA_BOOL8_CONSTRUCTOR(a / b.x,
                                a / b.y,
                                a / b.z,
                                a / b.w,
                                a / b.s4,
                                a / b.s5,
                                a / b.s6,
                                a / b.s7);
}

occaFunction inline bool8  operator /  (const bool8 &a, const bool &b){
  return OCCA_BOOL8_CONSTRUCTOR(a.x / b,
                                a.y / b,
                                a.z / b,
                                a.w / b,
                                a.s4 / b,
                                a.s5 / b,
                                a.s6 / b,
                                a.s7 / b);
}

occaFunction inline bool8& operator /= (      bool8 &a, const bool8 &b){
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

occaFunction inline bool8& operator /= (      bool8 &a, const bool &b){
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
inline std::ostream& operator << (std::ostream &out, const bool8& a){
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
#  define OCCA_BOOL16_CONSTRUCTOR bool16
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

occaFunction inline bool16  operator +  (const bool16 &a, const bool16 &b){
  return OCCA_BOOL16_CONSTRUCTOR(a.x + b.x,
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

occaFunction inline bool16  operator +  (const bool &a, const bool16 &b){
  return OCCA_BOOL16_CONSTRUCTOR(a + b.x,
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

occaFunction inline bool16  operator +  (const bool16 &a, const bool &b){
  return OCCA_BOOL16_CONSTRUCTOR(a.x + b,
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

occaFunction inline bool16& operator += (      bool16 &a, const bool16 &b){
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

occaFunction inline bool16& operator += (      bool16 &a, const bool &b){
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
occaFunction inline bool16  operator -  (const bool16 &a, const bool16 &b){
  return OCCA_BOOL16_CONSTRUCTOR(a.x - b.x,
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

occaFunction inline bool16  operator -  (const bool &a, const bool16 &b){
  return OCCA_BOOL16_CONSTRUCTOR(a - b.x,
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

occaFunction inline bool16  operator -  (const bool16 &a, const bool &b){
  return OCCA_BOOL16_CONSTRUCTOR(a.x - b,
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

occaFunction inline bool16& operator -= (      bool16 &a, const bool16 &b){
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

occaFunction inline bool16& operator -= (      bool16 &a, const bool &b){
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
occaFunction inline bool16  operator *  (const bool16 &a, const bool16 &b){
  return OCCA_BOOL16_CONSTRUCTOR(a.x * b.x,
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

occaFunction inline bool16  operator *  (const bool &a, const bool16 &b){
  return OCCA_BOOL16_CONSTRUCTOR(a * b.x,
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

occaFunction inline bool16  operator *  (const bool16 &a, const bool &b){
  return OCCA_BOOL16_CONSTRUCTOR(a.x * b,
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

occaFunction inline bool16& operator *= (      bool16 &a, const bool16 &b){
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

occaFunction inline bool16& operator *= (      bool16 &a, const bool &b){
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
occaFunction inline bool16  operator /  (const bool16 &a, const bool16 &b){
  return OCCA_BOOL16_CONSTRUCTOR(a.x / b.x,
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

occaFunction inline bool16  operator /  (const bool &a, const bool16 &b){
  return OCCA_BOOL16_CONSTRUCTOR(a / b.x,
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

occaFunction inline bool16  operator /  (const bool16 &a, const bool &b){
  return OCCA_BOOL16_CONSTRUCTOR(a.x / b,
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

occaFunction inline bool16& operator /= (      bool16 &a, const bool16 &b){
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

occaFunction inline bool16& operator /= (      bool16 &a, const bool &b){
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
inline std::ostream& operator << (std::ostream &out, const bool16& a){
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
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
#  define OCCA_CHAR2_CONSTRUCTOR char2
#else
#  define OCCA_CHAR2_CONSTRUCTOR make_char2
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

occaFunction inline char2 operator + (const char2 &a){
  return OCCA_CHAR2_CONSTRUCTOR(+a.x,
                                +a.y);
}

occaFunction inline char2 operator ++ (char2 &a, int){
  return OCCA_CHAR2_CONSTRUCTOR(a.x++,
                                a.y++);
}

occaFunction inline char2& operator ++ (char2 &a){
  ++a.x;
  ++a.y;
  return a;
}
occaFunction inline char2 operator - (const char2 &a){
  return OCCA_CHAR2_CONSTRUCTOR(-a.x,
                                -a.y);
}

occaFunction inline char2 operator -- (char2 &a, int){
  return OCCA_CHAR2_CONSTRUCTOR(a.x--,
                                a.y--);
}

occaFunction inline char2& operator -- (char2 &a){
  --a.x;
  --a.y;
  return a;
}
occaFunction inline char2  operator +  (const char2 &a, const char2 &b){
  return OCCA_CHAR2_CONSTRUCTOR(a.x + b.x,
                                a.y + b.y);
}

occaFunction inline char2  operator +  (const char &a, const char2 &b){
  return OCCA_CHAR2_CONSTRUCTOR(a + b.x,
                                a + b.y);
}

occaFunction inline char2  operator +  (const char2 &a, const char &b){
  return OCCA_CHAR2_CONSTRUCTOR(a.x + b,
                                a.y + b);
}

occaFunction inline char2& operator += (      char2 &a, const char2 &b){
  a.x += b.x;
  a.y += b.y;
  return a;
}

occaFunction inline char2& operator += (      char2 &a, const char &b){
  a.x += b;
  a.y += b;
  return a;
}
occaFunction inline char2  operator -  (const char2 &a, const char2 &b){
  return OCCA_CHAR2_CONSTRUCTOR(a.x - b.x,
                                a.y - b.y);
}

occaFunction inline char2  operator -  (const char &a, const char2 &b){
  return OCCA_CHAR2_CONSTRUCTOR(a - b.x,
                                a - b.y);
}

occaFunction inline char2  operator -  (const char2 &a, const char &b){
  return OCCA_CHAR2_CONSTRUCTOR(a.x - b,
                                a.y - b);
}

occaFunction inline char2& operator -= (      char2 &a, const char2 &b){
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

occaFunction inline char2& operator -= (      char2 &a, const char &b){
  a.x -= b;
  a.y -= b;
  return a;
}
occaFunction inline char2  operator *  (const char2 &a, const char2 &b){
  return OCCA_CHAR2_CONSTRUCTOR(a.x * b.x,
                                a.y * b.y);
}

occaFunction inline char2  operator *  (const char &a, const char2 &b){
  return OCCA_CHAR2_CONSTRUCTOR(a * b.x,
                                a * b.y);
}

occaFunction inline char2  operator *  (const char2 &a, const char &b){
  return OCCA_CHAR2_CONSTRUCTOR(a.x * b,
                                a.y * b);
}

occaFunction inline char2& operator *= (      char2 &a, const char2 &b){
  a.x *= b.x;
  a.y *= b.y;
  return a;
}

occaFunction inline char2& operator *= (      char2 &a, const char &b){
  a.x *= b;
  a.y *= b;
  return a;
}
occaFunction inline char2  operator /  (const char2 &a, const char2 &b){
  return OCCA_CHAR2_CONSTRUCTOR(a.x / b.x,
                                a.y / b.y);
}

occaFunction inline char2  operator /  (const char &a, const char2 &b){
  return OCCA_CHAR2_CONSTRUCTOR(a / b.x,
                                a / b.y);
}

occaFunction inline char2  operator /  (const char2 &a, const char &b){
  return OCCA_CHAR2_CONSTRUCTOR(a.x / b,
                                a.y / b);
}

occaFunction inline char2& operator /= (      char2 &a, const char2 &b){
  a.x /= b.x;
  a.y /= b.y;
  return a;
}

occaFunction inline char2& operator /= (      char2 &a, const char &b){
  a.x /= b;
  a.y /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const char2& a){
  out << "[" << a.x << ", "
             << a.y
      << "]\n";

  return out;
}
#endif

//======================================


//---[ char4 ]--------------------------
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
#  define OCCA_CHAR4_CONSTRUCTOR char4
#else
#  define OCCA_CHAR4_CONSTRUCTOR make_char4
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

occaFunction inline char4 operator + (const char4 &a){
  return OCCA_CHAR4_CONSTRUCTOR(+a.x,
                                +a.y,
                                +a.z,
                                +a.w);
}

occaFunction inline char4 operator ++ (char4 &a, int){
  return OCCA_CHAR4_CONSTRUCTOR(a.x++,
                                a.y++,
                                a.z++,
                                a.w++);
}

occaFunction inline char4& operator ++ (char4 &a){
  ++a.x;
  ++a.y;
  ++a.z;
  ++a.w;
  return a;
}
occaFunction inline char4 operator - (const char4 &a){
  return OCCA_CHAR4_CONSTRUCTOR(-a.x,
                                -a.y,
                                -a.z,
                                -a.w);
}

occaFunction inline char4 operator -- (char4 &a, int){
  return OCCA_CHAR4_CONSTRUCTOR(a.x--,
                                a.y--,
                                a.z--,
                                a.w--);
}

occaFunction inline char4& operator -- (char4 &a){
  --a.x;
  --a.y;
  --a.z;
  --a.w;
  return a;
}
occaFunction inline char4  operator +  (const char4 &a, const char4 &b){
  return OCCA_CHAR4_CONSTRUCTOR(a.x + b.x,
                                a.y + b.y,
                                a.z + b.z,
                                a.w + b.w);
}

occaFunction inline char4  operator +  (const char &a, const char4 &b){
  return OCCA_CHAR4_CONSTRUCTOR(a + b.x,
                                a + b.y,
                                a + b.z,
                                a + b.w);
}

occaFunction inline char4  operator +  (const char4 &a, const char &b){
  return OCCA_CHAR4_CONSTRUCTOR(a.x + b,
                                a.y + b,
                                a.z + b,
                                a.w + b);
}

occaFunction inline char4& operator += (      char4 &a, const char4 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

occaFunction inline char4& operator += (      char4 &a, const char &b){
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  return a;
}
occaFunction inline char4  operator -  (const char4 &a, const char4 &b){
  return OCCA_CHAR4_CONSTRUCTOR(a.x - b.x,
                                a.y - b.y,
                                a.z - b.z,
                                a.w - b.w);
}

occaFunction inline char4  operator -  (const char &a, const char4 &b){
  return OCCA_CHAR4_CONSTRUCTOR(a - b.x,
                                a - b.y,
                                a - b.z,
                                a - b.w);
}

occaFunction inline char4  operator -  (const char4 &a, const char &b){
  return OCCA_CHAR4_CONSTRUCTOR(a.x - b,
                                a.y - b,
                                a.z - b,
                                a.w - b);
}

occaFunction inline char4& operator -= (      char4 &a, const char4 &b){
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  return a;
}

occaFunction inline char4& operator -= (      char4 &a, const char &b){
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  return a;
}
occaFunction inline char4  operator *  (const char4 &a, const char4 &b){
  return OCCA_CHAR4_CONSTRUCTOR(a.x * b.x,
                                a.y * b.y,
                                a.z * b.z,
                                a.w * b.w);
}

occaFunction inline char4  operator *  (const char &a, const char4 &b){
  return OCCA_CHAR4_CONSTRUCTOR(a * b.x,
                                a * b.y,
                                a * b.z,
                                a * b.w);
}

occaFunction inline char4  operator *  (const char4 &a, const char &b){
  return OCCA_CHAR4_CONSTRUCTOR(a.x * b,
                                a.y * b,
                                a.z * b,
                                a.w * b);
}

occaFunction inline char4& operator *= (      char4 &a, const char4 &b){
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  return a;
}

occaFunction inline char4& operator *= (      char4 &a, const char &b){
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}
occaFunction inline char4  operator /  (const char4 &a, const char4 &b){
  return OCCA_CHAR4_CONSTRUCTOR(a.x / b.x,
                                a.y / b.y,
                                a.z / b.z,
                                a.w / b.w);
}

occaFunction inline char4  operator /  (const char &a, const char4 &b){
  return OCCA_CHAR4_CONSTRUCTOR(a / b.x,
                                a / b.y,
                                a / b.z,
                                a / b.w);
}

occaFunction inline char4  operator /  (const char4 &a, const char &b){
  return OCCA_CHAR4_CONSTRUCTOR(a.x / b,
                                a.y / b,
                                a.z / b,
                                a.w / b);
}

occaFunction inline char4& operator /= (      char4 &a, const char4 &b){
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  return a;
}

occaFunction inline char4& operator /= (      char4 &a, const char &b){
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const char4& a){
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
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
typedef char4 char3;
#endif
//======================================


//---[ char8 ]--------------------------
#  define OCCA_CHAR8_CONSTRUCTOR char8
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

occaFunction inline char8 operator + (const char8 &a){
  return OCCA_CHAR8_CONSTRUCTOR(+a.x,
                                +a.y,
                                +a.z,
                                +a.w,
                                +a.s4,
                                +a.s5,
                                +a.s6,
                                +a.s7);
}

occaFunction inline char8 operator ++ (char8 &a, int){
  return OCCA_CHAR8_CONSTRUCTOR(a.x++,
                                a.y++,
                                a.z++,
                                a.w++,
                                a.s4++,
                                a.s5++,
                                a.s6++,
                                a.s7++);
}

occaFunction inline char8& operator ++ (char8 &a){
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
occaFunction inline char8 operator - (const char8 &a){
  return OCCA_CHAR8_CONSTRUCTOR(-a.x,
                                -a.y,
                                -a.z,
                                -a.w,
                                -a.s4,
                                -a.s5,
                                -a.s6,
                                -a.s7);
}

occaFunction inline char8 operator -- (char8 &a, int){
  return OCCA_CHAR8_CONSTRUCTOR(a.x--,
                                a.y--,
                                a.z--,
                                a.w--,
                                a.s4--,
                                a.s5--,
                                a.s6--,
                                a.s7--);
}

occaFunction inline char8& operator -- (char8 &a){
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
occaFunction inline char8  operator +  (const char8 &a, const char8 &b){
  return OCCA_CHAR8_CONSTRUCTOR(a.x + b.x,
                                a.y + b.y,
                                a.z + b.z,
                                a.w + b.w,
                                a.s4 + b.s4,
                                a.s5 + b.s5,
                                a.s6 + b.s6,
                                a.s7 + b.s7);
}

occaFunction inline char8  operator +  (const char &a, const char8 &b){
  return OCCA_CHAR8_CONSTRUCTOR(a + b.x,
                                a + b.y,
                                a + b.z,
                                a + b.w,
                                a + b.s4,
                                a + b.s5,
                                a + b.s6,
                                a + b.s7);
}

occaFunction inline char8  operator +  (const char8 &a, const char &b){
  return OCCA_CHAR8_CONSTRUCTOR(a.x + b,
                                a.y + b,
                                a.z + b,
                                a.w + b,
                                a.s4 + b,
                                a.s5 + b,
                                a.s6 + b,
                                a.s7 + b);
}

occaFunction inline char8& operator += (      char8 &a, const char8 &b){
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

occaFunction inline char8& operator += (      char8 &a, const char &b){
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
occaFunction inline char8  operator -  (const char8 &a, const char8 &b){
  return OCCA_CHAR8_CONSTRUCTOR(a.x - b.x,
                                a.y - b.y,
                                a.z - b.z,
                                a.w - b.w,
                                a.s4 - b.s4,
                                a.s5 - b.s5,
                                a.s6 - b.s6,
                                a.s7 - b.s7);
}

occaFunction inline char8  operator -  (const char &a, const char8 &b){
  return OCCA_CHAR8_CONSTRUCTOR(a - b.x,
                                a - b.y,
                                a - b.z,
                                a - b.w,
                                a - b.s4,
                                a - b.s5,
                                a - b.s6,
                                a - b.s7);
}

occaFunction inline char8  operator -  (const char8 &a, const char &b){
  return OCCA_CHAR8_CONSTRUCTOR(a.x - b,
                                a.y - b,
                                a.z - b,
                                a.w - b,
                                a.s4 - b,
                                a.s5 - b,
                                a.s6 - b,
                                a.s7 - b);
}

occaFunction inline char8& operator -= (      char8 &a, const char8 &b){
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

occaFunction inline char8& operator -= (      char8 &a, const char &b){
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
occaFunction inline char8  operator *  (const char8 &a, const char8 &b){
  return OCCA_CHAR8_CONSTRUCTOR(a.x * b.x,
                                a.y * b.y,
                                a.z * b.z,
                                a.w * b.w,
                                a.s4 * b.s4,
                                a.s5 * b.s5,
                                a.s6 * b.s6,
                                a.s7 * b.s7);
}

occaFunction inline char8  operator *  (const char &a, const char8 &b){
  return OCCA_CHAR8_CONSTRUCTOR(a * b.x,
                                a * b.y,
                                a * b.z,
                                a * b.w,
                                a * b.s4,
                                a * b.s5,
                                a * b.s6,
                                a * b.s7);
}

occaFunction inline char8  operator *  (const char8 &a, const char &b){
  return OCCA_CHAR8_CONSTRUCTOR(a.x * b,
                                a.y * b,
                                a.z * b,
                                a.w * b,
                                a.s4 * b,
                                a.s5 * b,
                                a.s6 * b,
                                a.s7 * b);
}

occaFunction inline char8& operator *= (      char8 &a, const char8 &b){
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

occaFunction inline char8& operator *= (      char8 &a, const char &b){
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
occaFunction inline char8  operator /  (const char8 &a, const char8 &b){
  return OCCA_CHAR8_CONSTRUCTOR(a.x / b.x,
                                a.y / b.y,
                                a.z / b.z,
                                a.w / b.w,
                                a.s4 / b.s4,
                                a.s5 / b.s5,
                                a.s6 / b.s6,
                                a.s7 / b.s7);
}

occaFunction inline char8  operator /  (const char &a, const char8 &b){
  return OCCA_CHAR8_CONSTRUCTOR(a / b.x,
                                a / b.y,
                                a / b.z,
                                a / b.w,
                                a / b.s4,
                                a / b.s5,
                                a / b.s6,
                                a / b.s7);
}

occaFunction inline char8  operator /  (const char8 &a, const char &b){
  return OCCA_CHAR8_CONSTRUCTOR(a.x / b,
                                a.y / b,
                                a.z / b,
                                a.w / b,
                                a.s4 / b,
                                a.s5 / b,
                                a.s6 / b,
                                a.s7 / b);
}

occaFunction inline char8& operator /= (      char8 &a, const char8 &b){
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

occaFunction inline char8& operator /= (      char8 &a, const char &b){
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
inline std::ostream& operator << (std::ostream &out, const char8& a){
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
#  define OCCA_CHAR16_CONSTRUCTOR char16
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

occaFunction inline char16 operator + (const char16 &a){
  return OCCA_CHAR16_CONSTRUCTOR(+a.x,
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

occaFunction inline char16 operator ++ (char16 &a, int){
  return OCCA_CHAR16_CONSTRUCTOR(a.x++,
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

occaFunction inline char16& operator ++ (char16 &a){
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
occaFunction inline char16 operator - (const char16 &a){
  return OCCA_CHAR16_CONSTRUCTOR(-a.x,
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

occaFunction inline char16 operator -- (char16 &a, int){
  return OCCA_CHAR16_CONSTRUCTOR(a.x--,
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

occaFunction inline char16& operator -- (char16 &a){
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
occaFunction inline char16  operator +  (const char16 &a, const char16 &b){
  return OCCA_CHAR16_CONSTRUCTOR(a.x + b.x,
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

occaFunction inline char16  operator +  (const char &a, const char16 &b){
  return OCCA_CHAR16_CONSTRUCTOR(a + b.x,
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

occaFunction inline char16  operator +  (const char16 &a, const char &b){
  return OCCA_CHAR16_CONSTRUCTOR(a.x + b,
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

occaFunction inline char16& operator += (      char16 &a, const char16 &b){
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

occaFunction inline char16& operator += (      char16 &a, const char &b){
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
occaFunction inline char16  operator -  (const char16 &a, const char16 &b){
  return OCCA_CHAR16_CONSTRUCTOR(a.x - b.x,
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

occaFunction inline char16  operator -  (const char &a, const char16 &b){
  return OCCA_CHAR16_CONSTRUCTOR(a - b.x,
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

occaFunction inline char16  operator -  (const char16 &a, const char &b){
  return OCCA_CHAR16_CONSTRUCTOR(a.x - b,
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

occaFunction inline char16& operator -= (      char16 &a, const char16 &b){
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

occaFunction inline char16& operator -= (      char16 &a, const char &b){
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
occaFunction inline char16  operator *  (const char16 &a, const char16 &b){
  return OCCA_CHAR16_CONSTRUCTOR(a.x * b.x,
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

occaFunction inline char16  operator *  (const char &a, const char16 &b){
  return OCCA_CHAR16_CONSTRUCTOR(a * b.x,
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

occaFunction inline char16  operator *  (const char16 &a, const char &b){
  return OCCA_CHAR16_CONSTRUCTOR(a.x * b,
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

occaFunction inline char16& operator *= (      char16 &a, const char16 &b){
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

occaFunction inline char16& operator *= (      char16 &a, const char &b){
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
occaFunction inline char16  operator /  (const char16 &a, const char16 &b){
  return OCCA_CHAR16_CONSTRUCTOR(a.x / b.x,
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

occaFunction inline char16  operator /  (const char &a, const char16 &b){
  return OCCA_CHAR16_CONSTRUCTOR(a / b.x,
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

occaFunction inline char16  operator /  (const char16 &a, const char &b){
  return OCCA_CHAR16_CONSTRUCTOR(a.x / b,
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

occaFunction inline char16& operator /= (      char16 &a, const char16 &b){
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

occaFunction inline char16& operator /= (      char16 &a, const char &b){
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
inline std::ostream& operator << (std::ostream &out, const char16& a){
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
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
#  define OCCA_SHORT2_CONSTRUCTOR short2
#else
#  define OCCA_SHORT2_CONSTRUCTOR make_short2
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

occaFunction inline short2 operator + (const short2 &a){
  return OCCA_SHORT2_CONSTRUCTOR(+a.x,
                                 +a.y);
}

occaFunction inline short2 operator ++ (short2 &a, int){
  return OCCA_SHORT2_CONSTRUCTOR(a.x++,
                                 a.y++);
}

occaFunction inline short2& operator ++ (short2 &a){
  ++a.x;
  ++a.y;
  return a;
}
occaFunction inline short2 operator - (const short2 &a){
  return OCCA_SHORT2_CONSTRUCTOR(-a.x,
                                 -a.y);
}

occaFunction inline short2 operator -- (short2 &a, int){
  return OCCA_SHORT2_CONSTRUCTOR(a.x--,
                                 a.y--);
}

occaFunction inline short2& operator -- (short2 &a){
  --a.x;
  --a.y;
  return a;
}
occaFunction inline short2  operator +  (const short2 &a, const short2 &b){
  return OCCA_SHORT2_CONSTRUCTOR(a.x + b.x,
                                 a.y + b.y);
}

occaFunction inline short2  operator +  (const short &a, const short2 &b){
  return OCCA_SHORT2_CONSTRUCTOR(a + b.x,
                                 a + b.y);
}

occaFunction inline short2  operator +  (const short2 &a, const short &b){
  return OCCA_SHORT2_CONSTRUCTOR(a.x + b,
                                 a.y + b);
}

occaFunction inline short2& operator += (      short2 &a, const short2 &b){
  a.x += b.x;
  a.y += b.y;
  return a;
}

occaFunction inline short2& operator += (      short2 &a, const short &b){
  a.x += b;
  a.y += b;
  return a;
}
occaFunction inline short2  operator -  (const short2 &a, const short2 &b){
  return OCCA_SHORT2_CONSTRUCTOR(a.x - b.x,
                                 a.y - b.y);
}

occaFunction inline short2  operator -  (const short &a, const short2 &b){
  return OCCA_SHORT2_CONSTRUCTOR(a - b.x,
                                 a - b.y);
}

occaFunction inline short2  operator -  (const short2 &a, const short &b){
  return OCCA_SHORT2_CONSTRUCTOR(a.x - b,
                                 a.y - b);
}

occaFunction inline short2& operator -= (      short2 &a, const short2 &b){
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

occaFunction inline short2& operator -= (      short2 &a, const short &b){
  a.x -= b;
  a.y -= b;
  return a;
}
occaFunction inline short2  operator *  (const short2 &a, const short2 &b){
  return OCCA_SHORT2_CONSTRUCTOR(a.x * b.x,
                                 a.y * b.y);
}

occaFunction inline short2  operator *  (const short &a, const short2 &b){
  return OCCA_SHORT2_CONSTRUCTOR(a * b.x,
                                 a * b.y);
}

occaFunction inline short2  operator *  (const short2 &a, const short &b){
  return OCCA_SHORT2_CONSTRUCTOR(a.x * b,
                                 a.y * b);
}

occaFunction inline short2& operator *= (      short2 &a, const short2 &b){
  a.x *= b.x;
  a.y *= b.y;
  return a;
}

occaFunction inline short2& operator *= (      short2 &a, const short &b){
  a.x *= b;
  a.y *= b;
  return a;
}
occaFunction inline short2  operator /  (const short2 &a, const short2 &b){
  return OCCA_SHORT2_CONSTRUCTOR(a.x / b.x,
                                 a.y / b.y);
}

occaFunction inline short2  operator /  (const short &a, const short2 &b){
  return OCCA_SHORT2_CONSTRUCTOR(a / b.x,
                                 a / b.y);
}

occaFunction inline short2  operator /  (const short2 &a, const short &b){
  return OCCA_SHORT2_CONSTRUCTOR(a.x / b,
                                 a.y / b);
}

occaFunction inline short2& operator /= (      short2 &a, const short2 &b){
  a.x /= b.x;
  a.y /= b.y;
  return a;
}

occaFunction inline short2& operator /= (      short2 &a, const short &b){
  a.x /= b;
  a.y /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const short2& a){
  out << "[" << a.x << ", "
             << a.y
      << "]\n";

  return out;
}
#endif

//======================================


//---[ short4 ]-------------------------
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
#  define OCCA_SHORT4_CONSTRUCTOR short4
#else
#  define OCCA_SHORT4_CONSTRUCTOR make_short4
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

occaFunction inline short4 operator + (const short4 &a){
  return OCCA_SHORT4_CONSTRUCTOR(+a.x,
                                 +a.y,
                                 +a.z,
                                 +a.w);
}

occaFunction inline short4 operator ++ (short4 &a, int){
  return OCCA_SHORT4_CONSTRUCTOR(a.x++,
                                 a.y++,
                                 a.z++,
                                 a.w++);
}

occaFunction inline short4& operator ++ (short4 &a){
  ++a.x;
  ++a.y;
  ++a.z;
  ++a.w;
  return a;
}
occaFunction inline short4 operator - (const short4 &a){
  return OCCA_SHORT4_CONSTRUCTOR(-a.x,
                                 -a.y,
                                 -a.z,
                                 -a.w);
}

occaFunction inline short4 operator -- (short4 &a, int){
  return OCCA_SHORT4_CONSTRUCTOR(a.x--,
                                 a.y--,
                                 a.z--,
                                 a.w--);
}

occaFunction inline short4& operator -- (short4 &a){
  --a.x;
  --a.y;
  --a.z;
  --a.w;
  return a;
}
occaFunction inline short4  operator +  (const short4 &a, const short4 &b){
  return OCCA_SHORT4_CONSTRUCTOR(a.x + b.x,
                                 a.y + b.y,
                                 a.z + b.z,
                                 a.w + b.w);
}

occaFunction inline short4  operator +  (const short &a, const short4 &b){
  return OCCA_SHORT4_CONSTRUCTOR(a + b.x,
                                 a + b.y,
                                 a + b.z,
                                 a + b.w);
}

occaFunction inline short4  operator +  (const short4 &a, const short &b){
  return OCCA_SHORT4_CONSTRUCTOR(a.x + b,
                                 a.y + b,
                                 a.z + b,
                                 a.w + b);
}

occaFunction inline short4& operator += (      short4 &a, const short4 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

occaFunction inline short4& operator += (      short4 &a, const short &b){
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  return a;
}
occaFunction inline short4  operator -  (const short4 &a, const short4 &b){
  return OCCA_SHORT4_CONSTRUCTOR(a.x - b.x,
                                 a.y - b.y,
                                 a.z - b.z,
                                 a.w - b.w);
}

occaFunction inline short4  operator -  (const short &a, const short4 &b){
  return OCCA_SHORT4_CONSTRUCTOR(a - b.x,
                                 a - b.y,
                                 a - b.z,
                                 a - b.w);
}

occaFunction inline short4  operator -  (const short4 &a, const short &b){
  return OCCA_SHORT4_CONSTRUCTOR(a.x - b,
                                 a.y - b,
                                 a.z - b,
                                 a.w - b);
}

occaFunction inline short4& operator -= (      short4 &a, const short4 &b){
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  return a;
}

occaFunction inline short4& operator -= (      short4 &a, const short &b){
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  return a;
}
occaFunction inline short4  operator *  (const short4 &a, const short4 &b){
  return OCCA_SHORT4_CONSTRUCTOR(a.x * b.x,
                                 a.y * b.y,
                                 a.z * b.z,
                                 a.w * b.w);
}

occaFunction inline short4  operator *  (const short &a, const short4 &b){
  return OCCA_SHORT4_CONSTRUCTOR(a * b.x,
                                 a * b.y,
                                 a * b.z,
                                 a * b.w);
}

occaFunction inline short4  operator *  (const short4 &a, const short &b){
  return OCCA_SHORT4_CONSTRUCTOR(a.x * b,
                                 a.y * b,
                                 a.z * b,
                                 a.w * b);
}

occaFunction inline short4& operator *= (      short4 &a, const short4 &b){
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  return a;
}

occaFunction inline short4& operator *= (      short4 &a, const short &b){
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}
occaFunction inline short4  operator /  (const short4 &a, const short4 &b){
  return OCCA_SHORT4_CONSTRUCTOR(a.x / b.x,
                                 a.y / b.y,
                                 a.z / b.z,
                                 a.w / b.w);
}

occaFunction inline short4  operator /  (const short &a, const short4 &b){
  return OCCA_SHORT4_CONSTRUCTOR(a / b.x,
                                 a / b.y,
                                 a / b.z,
                                 a / b.w);
}

occaFunction inline short4  operator /  (const short4 &a, const short &b){
  return OCCA_SHORT4_CONSTRUCTOR(a.x / b,
                                 a.y / b,
                                 a.z / b,
                                 a.w / b);
}

occaFunction inline short4& operator /= (      short4 &a, const short4 &b){
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  return a;
}

occaFunction inline short4& operator /= (      short4 &a, const short &b){
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const short4& a){
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
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
typedef short4 short3;
#endif
//======================================


//---[ short8 ]-------------------------
#  define OCCA_SHORT8_CONSTRUCTOR short8
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

occaFunction inline short8 operator + (const short8 &a){
  return OCCA_SHORT8_CONSTRUCTOR(+a.x,
                                 +a.y,
                                 +a.z,
                                 +a.w,
                                 +a.s4,
                                 +a.s5,
                                 +a.s6,
                                 +a.s7);
}

occaFunction inline short8 operator ++ (short8 &a, int){
  return OCCA_SHORT8_CONSTRUCTOR(a.x++,
                                 a.y++,
                                 a.z++,
                                 a.w++,
                                 a.s4++,
                                 a.s5++,
                                 a.s6++,
                                 a.s7++);
}

occaFunction inline short8& operator ++ (short8 &a){
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
occaFunction inline short8 operator - (const short8 &a){
  return OCCA_SHORT8_CONSTRUCTOR(-a.x,
                                 -a.y,
                                 -a.z,
                                 -a.w,
                                 -a.s4,
                                 -a.s5,
                                 -a.s6,
                                 -a.s7);
}

occaFunction inline short8 operator -- (short8 &a, int){
  return OCCA_SHORT8_CONSTRUCTOR(a.x--,
                                 a.y--,
                                 a.z--,
                                 a.w--,
                                 a.s4--,
                                 a.s5--,
                                 a.s6--,
                                 a.s7--);
}

occaFunction inline short8& operator -- (short8 &a){
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
occaFunction inline short8  operator +  (const short8 &a, const short8 &b){
  return OCCA_SHORT8_CONSTRUCTOR(a.x + b.x,
                                 a.y + b.y,
                                 a.z + b.z,
                                 a.w + b.w,
                                 a.s4 + b.s4,
                                 a.s5 + b.s5,
                                 a.s6 + b.s6,
                                 a.s7 + b.s7);
}

occaFunction inline short8  operator +  (const short &a, const short8 &b){
  return OCCA_SHORT8_CONSTRUCTOR(a + b.x,
                                 a + b.y,
                                 a + b.z,
                                 a + b.w,
                                 a + b.s4,
                                 a + b.s5,
                                 a + b.s6,
                                 a + b.s7);
}

occaFunction inline short8  operator +  (const short8 &a, const short &b){
  return OCCA_SHORT8_CONSTRUCTOR(a.x + b,
                                 a.y + b,
                                 a.z + b,
                                 a.w + b,
                                 a.s4 + b,
                                 a.s5 + b,
                                 a.s6 + b,
                                 a.s7 + b);
}

occaFunction inline short8& operator += (      short8 &a, const short8 &b){
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

occaFunction inline short8& operator += (      short8 &a, const short &b){
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
occaFunction inline short8  operator -  (const short8 &a, const short8 &b){
  return OCCA_SHORT8_CONSTRUCTOR(a.x - b.x,
                                 a.y - b.y,
                                 a.z - b.z,
                                 a.w - b.w,
                                 a.s4 - b.s4,
                                 a.s5 - b.s5,
                                 a.s6 - b.s6,
                                 a.s7 - b.s7);
}

occaFunction inline short8  operator -  (const short &a, const short8 &b){
  return OCCA_SHORT8_CONSTRUCTOR(a - b.x,
                                 a - b.y,
                                 a - b.z,
                                 a - b.w,
                                 a - b.s4,
                                 a - b.s5,
                                 a - b.s6,
                                 a - b.s7);
}

occaFunction inline short8  operator -  (const short8 &a, const short &b){
  return OCCA_SHORT8_CONSTRUCTOR(a.x - b,
                                 a.y - b,
                                 a.z - b,
                                 a.w - b,
                                 a.s4 - b,
                                 a.s5 - b,
                                 a.s6 - b,
                                 a.s7 - b);
}

occaFunction inline short8& operator -= (      short8 &a, const short8 &b){
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

occaFunction inline short8& operator -= (      short8 &a, const short &b){
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
occaFunction inline short8  operator *  (const short8 &a, const short8 &b){
  return OCCA_SHORT8_CONSTRUCTOR(a.x * b.x,
                                 a.y * b.y,
                                 a.z * b.z,
                                 a.w * b.w,
                                 a.s4 * b.s4,
                                 a.s5 * b.s5,
                                 a.s6 * b.s6,
                                 a.s7 * b.s7);
}

occaFunction inline short8  operator *  (const short &a, const short8 &b){
  return OCCA_SHORT8_CONSTRUCTOR(a * b.x,
                                 a * b.y,
                                 a * b.z,
                                 a * b.w,
                                 a * b.s4,
                                 a * b.s5,
                                 a * b.s6,
                                 a * b.s7);
}

occaFunction inline short8  operator *  (const short8 &a, const short &b){
  return OCCA_SHORT8_CONSTRUCTOR(a.x * b,
                                 a.y * b,
                                 a.z * b,
                                 a.w * b,
                                 a.s4 * b,
                                 a.s5 * b,
                                 a.s6 * b,
                                 a.s7 * b);
}

occaFunction inline short8& operator *= (      short8 &a, const short8 &b){
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

occaFunction inline short8& operator *= (      short8 &a, const short &b){
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
occaFunction inline short8  operator /  (const short8 &a, const short8 &b){
  return OCCA_SHORT8_CONSTRUCTOR(a.x / b.x,
                                 a.y / b.y,
                                 a.z / b.z,
                                 a.w / b.w,
                                 a.s4 / b.s4,
                                 a.s5 / b.s5,
                                 a.s6 / b.s6,
                                 a.s7 / b.s7);
}

occaFunction inline short8  operator /  (const short &a, const short8 &b){
  return OCCA_SHORT8_CONSTRUCTOR(a / b.x,
                                 a / b.y,
                                 a / b.z,
                                 a / b.w,
                                 a / b.s4,
                                 a / b.s5,
                                 a / b.s6,
                                 a / b.s7);
}

occaFunction inline short8  operator /  (const short8 &a, const short &b){
  return OCCA_SHORT8_CONSTRUCTOR(a.x / b,
                                 a.y / b,
                                 a.z / b,
                                 a.w / b,
                                 a.s4 / b,
                                 a.s5 / b,
                                 a.s6 / b,
                                 a.s7 / b);
}

occaFunction inline short8& operator /= (      short8 &a, const short8 &b){
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

occaFunction inline short8& operator /= (      short8 &a, const short &b){
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
inline std::ostream& operator << (std::ostream &out, const short8& a){
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
#  define OCCA_SHORT16_CONSTRUCTOR short16
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

occaFunction inline short16 operator + (const short16 &a){
  return OCCA_SHORT16_CONSTRUCTOR(+a.x,
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

occaFunction inline short16 operator ++ (short16 &a, int){
  return OCCA_SHORT16_CONSTRUCTOR(a.x++,
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

occaFunction inline short16& operator ++ (short16 &a){
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
occaFunction inline short16 operator - (const short16 &a){
  return OCCA_SHORT16_CONSTRUCTOR(-a.x,
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

occaFunction inline short16 operator -- (short16 &a, int){
  return OCCA_SHORT16_CONSTRUCTOR(a.x--,
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

occaFunction inline short16& operator -- (short16 &a){
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
occaFunction inline short16  operator +  (const short16 &a, const short16 &b){
  return OCCA_SHORT16_CONSTRUCTOR(a.x + b.x,
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

occaFunction inline short16  operator +  (const short &a, const short16 &b){
  return OCCA_SHORT16_CONSTRUCTOR(a + b.x,
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

occaFunction inline short16  operator +  (const short16 &a, const short &b){
  return OCCA_SHORT16_CONSTRUCTOR(a.x + b,
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

occaFunction inline short16& operator += (      short16 &a, const short16 &b){
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

occaFunction inline short16& operator += (      short16 &a, const short &b){
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
occaFunction inline short16  operator -  (const short16 &a, const short16 &b){
  return OCCA_SHORT16_CONSTRUCTOR(a.x - b.x,
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

occaFunction inline short16  operator -  (const short &a, const short16 &b){
  return OCCA_SHORT16_CONSTRUCTOR(a - b.x,
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

occaFunction inline short16  operator -  (const short16 &a, const short &b){
  return OCCA_SHORT16_CONSTRUCTOR(a.x - b,
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

occaFunction inline short16& operator -= (      short16 &a, const short16 &b){
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

occaFunction inline short16& operator -= (      short16 &a, const short &b){
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
occaFunction inline short16  operator *  (const short16 &a, const short16 &b){
  return OCCA_SHORT16_CONSTRUCTOR(a.x * b.x,
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

occaFunction inline short16  operator *  (const short &a, const short16 &b){
  return OCCA_SHORT16_CONSTRUCTOR(a * b.x,
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

occaFunction inline short16  operator *  (const short16 &a, const short &b){
  return OCCA_SHORT16_CONSTRUCTOR(a.x * b,
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

occaFunction inline short16& operator *= (      short16 &a, const short16 &b){
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

occaFunction inline short16& operator *= (      short16 &a, const short &b){
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
occaFunction inline short16  operator /  (const short16 &a, const short16 &b){
  return OCCA_SHORT16_CONSTRUCTOR(a.x / b.x,
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

occaFunction inline short16  operator /  (const short &a, const short16 &b){
  return OCCA_SHORT16_CONSTRUCTOR(a / b.x,
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

occaFunction inline short16  operator /  (const short16 &a, const short &b){
  return OCCA_SHORT16_CONSTRUCTOR(a.x / b,
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

occaFunction inline short16& operator /= (      short16 &a, const short16 &b){
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

occaFunction inline short16& operator /= (      short16 &a, const short &b){
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
inline std::ostream& operator << (std::ostream &out, const short16& a){
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
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
#  define OCCA_INT2_CONSTRUCTOR int2
#else
#  define OCCA_INT2_CONSTRUCTOR make_int2
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

occaFunction inline int2 operator + (const int2 &a){
  return OCCA_INT2_CONSTRUCTOR(+a.x,
                               +a.y);
}

occaFunction inline int2 operator ++ (int2 &a, int){
  return OCCA_INT2_CONSTRUCTOR(a.x++,
                               a.y++);
}

occaFunction inline int2& operator ++ (int2 &a){
  ++a.x;
  ++a.y;
  return a;
}
occaFunction inline int2 operator - (const int2 &a){
  return OCCA_INT2_CONSTRUCTOR(-a.x,
                               -a.y);
}

occaFunction inline int2 operator -- (int2 &a, int){
  return OCCA_INT2_CONSTRUCTOR(a.x--,
                               a.y--);
}

occaFunction inline int2& operator -- (int2 &a){
  --a.x;
  --a.y;
  return a;
}
occaFunction inline int2  operator +  (const int2 &a, const int2 &b){
  return OCCA_INT2_CONSTRUCTOR(a.x + b.x,
                               a.y + b.y);
}

occaFunction inline int2  operator +  (const int &a, const int2 &b){
  return OCCA_INT2_CONSTRUCTOR(a + b.x,
                               a + b.y);
}

occaFunction inline int2  operator +  (const int2 &a, const int &b){
  return OCCA_INT2_CONSTRUCTOR(a.x + b,
                               a.y + b);
}

occaFunction inline int2& operator += (      int2 &a, const int2 &b){
  a.x += b.x;
  a.y += b.y;
  return a;
}

occaFunction inline int2& operator += (      int2 &a, const int &b){
  a.x += b;
  a.y += b;
  return a;
}
occaFunction inline int2  operator -  (const int2 &a, const int2 &b){
  return OCCA_INT2_CONSTRUCTOR(a.x - b.x,
                               a.y - b.y);
}

occaFunction inline int2  operator -  (const int &a, const int2 &b){
  return OCCA_INT2_CONSTRUCTOR(a - b.x,
                               a - b.y);
}

occaFunction inline int2  operator -  (const int2 &a, const int &b){
  return OCCA_INT2_CONSTRUCTOR(a.x - b,
                               a.y - b);
}

occaFunction inline int2& operator -= (      int2 &a, const int2 &b){
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

occaFunction inline int2& operator -= (      int2 &a, const int &b){
  a.x -= b;
  a.y -= b;
  return a;
}
occaFunction inline int2  operator *  (const int2 &a, const int2 &b){
  return OCCA_INT2_CONSTRUCTOR(a.x * b.x,
                               a.y * b.y);
}

occaFunction inline int2  operator *  (const int &a, const int2 &b){
  return OCCA_INT2_CONSTRUCTOR(a * b.x,
                               a * b.y);
}

occaFunction inline int2  operator *  (const int2 &a, const int &b){
  return OCCA_INT2_CONSTRUCTOR(a.x * b,
                               a.y * b);
}

occaFunction inline int2& operator *= (      int2 &a, const int2 &b){
  a.x *= b.x;
  a.y *= b.y;
  return a;
}

occaFunction inline int2& operator *= (      int2 &a, const int &b){
  a.x *= b;
  a.y *= b;
  return a;
}
occaFunction inline int2  operator /  (const int2 &a, const int2 &b){
  return OCCA_INT2_CONSTRUCTOR(a.x / b.x,
                               a.y / b.y);
}

occaFunction inline int2  operator /  (const int &a, const int2 &b){
  return OCCA_INT2_CONSTRUCTOR(a / b.x,
                               a / b.y);
}

occaFunction inline int2  operator /  (const int2 &a, const int &b){
  return OCCA_INT2_CONSTRUCTOR(a.x / b,
                               a.y / b);
}

occaFunction inline int2& operator /= (      int2 &a, const int2 &b){
  a.x /= b.x;
  a.y /= b.y;
  return a;
}

occaFunction inline int2& operator /= (      int2 &a, const int &b){
  a.x /= b;
  a.y /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const int2& a){
  out << "[" << a.x << ", "
             << a.y
      << "]\n";

  return out;
}
#endif

//======================================


//---[ int4 ]---------------------------
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
#  define OCCA_INT4_CONSTRUCTOR int4
#else
#  define OCCA_INT4_CONSTRUCTOR make_int4
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

occaFunction inline int4 operator + (const int4 &a){
  return OCCA_INT4_CONSTRUCTOR(+a.x,
                               +a.y,
                               +a.z,
                               +a.w);
}

occaFunction inline int4 operator ++ (int4 &a, int){
  return OCCA_INT4_CONSTRUCTOR(a.x++,
                               a.y++,
                               a.z++,
                               a.w++);
}

occaFunction inline int4& operator ++ (int4 &a){
  ++a.x;
  ++a.y;
  ++a.z;
  ++a.w;
  return a;
}
occaFunction inline int4 operator - (const int4 &a){
  return OCCA_INT4_CONSTRUCTOR(-a.x,
                               -a.y,
                               -a.z,
                               -a.w);
}

occaFunction inline int4 operator -- (int4 &a, int){
  return OCCA_INT4_CONSTRUCTOR(a.x--,
                               a.y--,
                               a.z--,
                               a.w--);
}

occaFunction inline int4& operator -- (int4 &a){
  --a.x;
  --a.y;
  --a.z;
  --a.w;
  return a;
}
occaFunction inline int4  operator +  (const int4 &a, const int4 &b){
  return OCCA_INT4_CONSTRUCTOR(a.x + b.x,
                               a.y + b.y,
                               a.z + b.z,
                               a.w + b.w);
}

occaFunction inline int4  operator +  (const int &a, const int4 &b){
  return OCCA_INT4_CONSTRUCTOR(a + b.x,
                               a + b.y,
                               a + b.z,
                               a + b.w);
}

occaFunction inline int4  operator +  (const int4 &a, const int &b){
  return OCCA_INT4_CONSTRUCTOR(a.x + b,
                               a.y + b,
                               a.z + b,
                               a.w + b);
}

occaFunction inline int4& operator += (      int4 &a, const int4 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

occaFunction inline int4& operator += (      int4 &a, const int &b){
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  return a;
}
occaFunction inline int4  operator -  (const int4 &a, const int4 &b){
  return OCCA_INT4_CONSTRUCTOR(a.x - b.x,
                               a.y - b.y,
                               a.z - b.z,
                               a.w - b.w);
}

occaFunction inline int4  operator -  (const int &a, const int4 &b){
  return OCCA_INT4_CONSTRUCTOR(a - b.x,
                               a - b.y,
                               a - b.z,
                               a - b.w);
}

occaFunction inline int4  operator -  (const int4 &a, const int &b){
  return OCCA_INT4_CONSTRUCTOR(a.x - b,
                               a.y - b,
                               a.z - b,
                               a.w - b);
}

occaFunction inline int4& operator -= (      int4 &a, const int4 &b){
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  return a;
}

occaFunction inline int4& operator -= (      int4 &a, const int &b){
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  return a;
}
occaFunction inline int4  operator *  (const int4 &a, const int4 &b){
  return OCCA_INT4_CONSTRUCTOR(a.x * b.x,
                               a.y * b.y,
                               a.z * b.z,
                               a.w * b.w);
}

occaFunction inline int4  operator *  (const int &a, const int4 &b){
  return OCCA_INT4_CONSTRUCTOR(a * b.x,
                               a * b.y,
                               a * b.z,
                               a * b.w);
}

occaFunction inline int4  operator *  (const int4 &a, const int &b){
  return OCCA_INT4_CONSTRUCTOR(a.x * b,
                               a.y * b,
                               a.z * b,
                               a.w * b);
}

occaFunction inline int4& operator *= (      int4 &a, const int4 &b){
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  return a;
}

occaFunction inline int4& operator *= (      int4 &a, const int &b){
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}
occaFunction inline int4  operator /  (const int4 &a, const int4 &b){
  return OCCA_INT4_CONSTRUCTOR(a.x / b.x,
                               a.y / b.y,
                               a.z / b.z,
                               a.w / b.w);
}

occaFunction inline int4  operator /  (const int &a, const int4 &b){
  return OCCA_INT4_CONSTRUCTOR(a / b.x,
                               a / b.y,
                               a / b.z,
                               a / b.w);
}

occaFunction inline int4  operator /  (const int4 &a, const int &b){
  return OCCA_INT4_CONSTRUCTOR(a.x / b,
                               a.y / b,
                               a.z / b,
                               a.w / b);
}

occaFunction inline int4& operator /= (      int4 &a, const int4 &b){
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  return a;
}

occaFunction inline int4& operator /= (      int4 &a, const int &b){
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const int4& a){
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
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
typedef int4 int3;
#endif
//======================================


//---[ int8 ]---------------------------
#  define OCCA_INT8_CONSTRUCTOR int8
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

occaFunction inline int8 operator + (const int8 &a){
  return OCCA_INT8_CONSTRUCTOR(+a.x,
                               +a.y,
                               +a.z,
                               +a.w,
                               +a.s4,
                               +a.s5,
                               +a.s6,
                               +a.s7);
}

occaFunction inline int8 operator ++ (int8 &a, int){
  return OCCA_INT8_CONSTRUCTOR(a.x++,
                               a.y++,
                               a.z++,
                               a.w++,
                               a.s4++,
                               a.s5++,
                               a.s6++,
                               a.s7++);
}

occaFunction inline int8& operator ++ (int8 &a){
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
occaFunction inline int8 operator - (const int8 &a){
  return OCCA_INT8_CONSTRUCTOR(-a.x,
                               -a.y,
                               -a.z,
                               -a.w,
                               -a.s4,
                               -a.s5,
                               -a.s6,
                               -a.s7);
}

occaFunction inline int8 operator -- (int8 &a, int){
  return OCCA_INT8_CONSTRUCTOR(a.x--,
                               a.y--,
                               a.z--,
                               a.w--,
                               a.s4--,
                               a.s5--,
                               a.s6--,
                               a.s7--);
}

occaFunction inline int8& operator -- (int8 &a){
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
occaFunction inline int8  operator +  (const int8 &a, const int8 &b){
  return OCCA_INT8_CONSTRUCTOR(a.x + b.x,
                               a.y + b.y,
                               a.z + b.z,
                               a.w + b.w,
                               a.s4 + b.s4,
                               a.s5 + b.s5,
                               a.s6 + b.s6,
                               a.s7 + b.s7);
}

occaFunction inline int8  operator +  (const int &a, const int8 &b){
  return OCCA_INT8_CONSTRUCTOR(a + b.x,
                               a + b.y,
                               a + b.z,
                               a + b.w,
                               a + b.s4,
                               a + b.s5,
                               a + b.s6,
                               a + b.s7);
}

occaFunction inline int8  operator +  (const int8 &a, const int &b){
  return OCCA_INT8_CONSTRUCTOR(a.x + b,
                               a.y + b,
                               a.z + b,
                               a.w + b,
                               a.s4 + b,
                               a.s5 + b,
                               a.s6 + b,
                               a.s7 + b);
}

occaFunction inline int8& operator += (      int8 &a, const int8 &b){
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

occaFunction inline int8& operator += (      int8 &a, const int &b){
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
occaFunction inline int8  operator -  (const int8 &a, const int8 &b){
  return OCCA_INT8_CONSTRUCTOR(a.x - b.x,
                               a.y - b.y,
                               a.z - b.z,
                               a.w - b.w,
                               a.s4 - b.s4,
                               a.s5 - b.s5,
                               a.s6 - b.s6,
                               a.s7 - b.s7);
}

occaFunction inline int8  operator -  (const int &a, const int8 &b){
  return OCCA_INT8_CONSTRUCTOR(a - b.x,
                               a - b.y,
                               a - b.z,
                               a - b.w,
                               a - b.s4,
                               a - b.s5,
                               a - b.s6,
                               a - b.s7);
}

occaFunction inline int8  operator -  (const int8 &a, const int &b){
  return OCCA_INT8_CONSTRUCTOR(a.x - b,
                               a.y - b,
                               a.z - b,
                               a.w - b,
                               a.s4 - b,
                               a.s5 - b,
                               a.s6 - b,
                               a.s7 - b);
}

occaFunction inline int8& operator -= (      int8 &a, const int8 &b){
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

occaFunction inline int8& operator -= (      int8 &a, const int &b){
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
occaFunction inline int8  operator *  (const int8 &a, const int8 &b){
  return OCCA_INT8_CONSTRUCTOR(a.x * b.x,
                               a.y * b.y,
                               a.z * b.z,
                               a.w * b.w,
                               a.s4 * b.s4,
                               a.s5 * b.s5,
                               a.s6 * b.s6,
                               a.s7 * b.s7);
}

occaFunction inline int8  operator *  (const int &a, const int8 &b){
  return OCCA_INT8_CONSTRUCTOR(a * b.x,
                               a * b.y,
                               a * b.z,
                               a * b.w,
                               a * b.s4,
                               a * b.s5,
                               a * b.s6,
                               a * b.s7);
}

occaFunction inline int8  operator *  (const int8 &a, const int &b){
  return OCCA_INT8_CONSTRUCTOR(a.x * b,
                               a.y * b,
                               a.z * b,
                               a.w * b,
                               a.s4 * b,
                               a.s5 * b,
                               a.s6 * b,
                               a.s7 * b);
}

occaFunction inline int8& operator *= (      int8 &a, const int8 &b){
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

occaFunction inline int8& operator *= (      int8 &a, const int &b){
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
occaFunction inline int8  operator /  (const int8 &a, const int8 &b){
  return OCCA_INT8_CONSTRUCTOR(a.x / b.x,
                               a.y / b.y,
                               a.z / b.z,
                               a.w / b.w,
                               a.s4 / b.s4,
                               a.s5 / b.s5,
                               a.s6 / b.s6,
                               a.s7 / b.s7);
}

occaFunction inline int8  operator /  (const int &a, const int8 &b){
  return OCCA_INT8_CONSTRUCTOR(a / b.x,
                               a / b.y,
                               a / b.z,
                               a / b.w,
                               a / b.s4,
                               a / b.s5,
                               a / b.s6,
                               a / b.s7);
}

occaFunction inline int8  operator /  (const int8 &a, const int &b){
  return OCCA_INT8_CONSTRUCTOR(a.x / b,
                               a.y / b,
                               a.z / b,
                               a.w / b,
                               a.s4 / b,
                               a.s5 / b,
                               a.s6 / b,
                               a.s7 / b);
}

occaFunction inline int8& operator /= (      int8 &a, const int8 &b){
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

occaFunction inline int8& operator /= (      int8 &a, const int &b){
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
inline std::ostream& operator << (std::ostream &out, const int8& a){
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
#  define OCCA_INT16_CONSTRUCTOR int16
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

occaFunction inline int16 operator + (const int16 &a){
  return OCCA_INT16_CONSTRUCTOR(+a.x,
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

occaFunction inline int16 operator ++ (int16 &a, int){
  return OCCA_INT16_CONSTRUCTOR(a.x++,
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

occaFunction inline int16& operator ++ (int16 &a){
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
occaFunction inline int16 operator - (const int16 &a){
  return OCCA_INT16_CONSTRUCTOR(-a.x,
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

occaFunction inline int16 operator -- (int16 &a, int){
  return OCCA_INT16_CONSTRUCTOR(a.x--,
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

occaFunction inline int16& operator -- (int16 &a){
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
occaFunction inline int16  operator +  (const int16 &a, const int16 &b){
  return OCCA_INT16_CONSTRUCTOR(a.x + b.x,
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

occaFunction inline int16  operator +  (const int &a, const int16 &b){
  return OCCA_INT16_CONSTRUCTOR(a + b.x,
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

occaFunction inline int16  operator +  (const int16 &a, const int &b){
  return OCCA_INT16_CONSTRUCTOR(a.x + b,
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

occaFunction inline int16& operator += (      int16 &a, const int16 &b){
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

occaFunction inline int16& operator += (      int16 &a, const int &b){
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
occaFunction inline int16  operator -  (const int16 &a, const int16 &b){
  return OCCA_INT16_CONSTRUCTOR(a.x - b.x,
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

occaFunction inline int16  operator -  (const int &a, const int16 &b){
  return OCCA_INT16_CONSTRUCTOR(a - b.x,
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

occaFunction inline int16  operator -  (const int16 &a, const int &b){
  return OCCA_INT16_CONSTRUCTOR(a.x - b,
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

occaFunction inline int16& operator -= (      int16 &a, const int16 &b){
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

occaFunction inline int16& operator -= (      int16 &a, const int &b){
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
occaFunction inline int16  operator *  (const int16 &a, const int16 &b){
  return OCCA_INT16_CONSTRUCTOR(a.x * b.x,
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

occaFunction inline int16  operator *  (const int &a, const int16 &b){
  return OCCA_INT16_CONSTRUCTOR(a * b.x,
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

occaFunction inline int16  operator *  (const int16 &a, const int &b){
  return OCCA_INT16_CONSTRUCTOR(a.x * b,
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

occaFunction inline int16& operator *= (      int16 &a, const int16 &b){
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

occaFunction inline int16& operator *= (      int16 &a, const int &b){
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
occaFunction inline int16  operator /  (const int16 &a, const int16 &b){
  return OCCA_INT16_CONSTRUCTOR(a.x / b.x,
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

occaFunction inline int16  operator /  (const int &a, const int16 &b){
  return OCCA_INT16_CONSTRUCTOR(a / b.x,
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

occaFunction inline int16  operator /  (const int16 &a, const int &b){
  return OCCA_INT16_CONSTRUCTOR(a.x / b,
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

occaFunction inline int16& operator /= (      int16 &a, const int16 &b){
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

occaFunction inline int16& operator /= (      int16 &a, const int &b){
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
inline std::ostream& operator << (std::ostream &out, const int16& a){
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
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
#  define OCCA_LONG2_CONSTRUCTOR long2
#else
#  define OCCA_LONG2_CONSTRUCTOR make_long2
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

occaFunction inline long2 operator + (const long2 &a){
  return OCCA_LONG2_CONSTRUCTOR(+a.x,
                                +a.y);
}

occaFunction inline long2 operator ++ (long2 &a, int){
  return OCCA_LONG2_CONSTRUCTOR(a.x++,
                                a.y++);
}

occaFunction inline long2& operator ++ (long2 &a){
  ++a.x;
  ++a.y;
  return a;
}
occaFunction inline long2 operator - (const long2 &a){
  return OCCA_LONG2_CONSTRUCTOR(-a.x,
                                -a.y);
}

occaFunction inline long2 operator -- (long2 &a, int){
  return OCCA_LONG2_CONSTRUCTOR(a.x--,
                                a.y--);
}

occaFunction inline long2& operator -- (long2 &a){
  --a.x;
  --a.y;
  return a;
}
occaFunction inline long2  operator +  (const long2 &a, const long2 &b){
  return OCCA_LONG2_CONSTRUCTOR(a.x + b.x,
                                a.y + b.y);
}

occaFunction inline long2  operator +  (const long &a, const long2 &b){
  return OCCA_LONG2_CONSTRUCTOR(a + b.x,
                                a + b.y);
}

occaFunction inline long2  operator +  (const long2 &a, const long &b){
  return OCCA_LONG2_CONSTRUCTOR(a.x + b,
                                a.y + b);
}

occaFunction inline long2& operator += (      long2 &a, const long2 &b){
  a.x += b.x;
  a.y += b.y;
  return a;
}

occaFunction inline long2& operator += (      long2 &a, const long &b){
  a.x += b;
  a.y += b;
  return a;
}
occaFunction inline long2  operator -  (const long2 &a, const long2 &b){
  return OCCA_LONG2_CONSTRUCTOR(a.x - b.x,
                                a.y - b.y);
}

occaFunction inline long2  operator -  (const long &a, const long2 &b){
  return OCCA_LONG2_CONSTRUCTOR(a - b.x,
                                a - b.y);
}

occaFunction inline long2  operator -  (const long2 &a, const long &b){
  return OCCA_LONG2_CONSTRUCTOR(a.x - b,
                                a.y - b);
}

occaFunction inline long2& operator -= (      long2 &a, const long2 &b){
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

occaFunction inline long2& operator -= (      long2 &a, const long &b){
  a.x -= b;
  a.y -= b;
  return a;
}
occaFunction inline long2  operator *  (const long2 &a, const long2 &b){
  return OCCA_LONG2_CONSTRUCTOR(a.x * b.x,
                                a.y * b.y);
}

occaFunction inline long2  operator *  (const long &a, const long2 &b){
  return OCCA_LONG2_CONSTRUCTOR(a * b.x,
                                a * b.y);
}

occaFunction inline long2  operator *  (const long2 &a, const long &b){
  return OCCA_LONG2_CONSTRUCTOR(a.x * b,
                                a.y * b);
}

occaFunction inline long2& operator *= (      long2 &a, const long2 &b){
  a.x *= b.x;
  a.y *= b.y;
  return a;
}

occaFunction inline long2& operator *= (      long2 &a, const long &b){
  a.x *= b;
  a.y *= b;
  return a;
}
occaFunction inline long2  operator /  (const long2 &a, const long2 &b){
  return OCCA_LONG2_CONSTRUCTOR(a.x / b.x,
                                a.y / b.y);
}

occaFunction inline long2  operator /  (const long &a, const long2 &b){
  return OCCA_LONG2_CONSTRUCTOR(a / b.x,
                                a / b.y);
}

occaFunction inline long2  operator /  (const long2 &a, const long &b){
  return OCCA_LONG2_CONSTRUCTOR(a.x / b,
                                a.y / b);
}

occaFunction inline long2& operator /= (      long2 &a, const long2 &b){
  a.x /= b.x;
  a.y /= b.y;
  return a;
}

occaFunction inline long2& operator /= (      long2 &a, const long &b){
  a.x /= b;
  a.y /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const long2& a){
  out << "[" << a.x << ", "
             << a.y
      << "]\n";

  return out;
}
#endif

//======================================


//---[ long4 ]--------------------------
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
#  define OCCA_LONG4_CONSTRUCTOR long4
#else
#  define OCCA_LONG4_CONSTRUCTOR make_long4
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

occaFunction inline long4 operator + (const long4 &a){
  return OCCA_LONG4_CONSTRUCTOR(+a.x,
                                +a.y,
                                +a.z,
                                +a.w);
}

occaFunction inline long4 operator ++ (long4 &a, int){
  return OCCA_LONG4_CONSTRUCTOR(a.x++,
                                a.y++,
                                a.z++,
                                a.w++);
}

occaFunction inline long4& operator ++ (long4 &a){
  ++a.x;
  ++a.y;
  ++a.z;
  ++a.w;
  return a;
}
occaFunction inline long4 operator - (const long4 &a){
  return OCCA_LONG4_CONSTRUCTOR(-a.x,
                                -a.y,
                                -a.z,
                                -a.w);
}

occaFunction inline long4 operator -- (long4 &a, int){
  return OCCA_LONG4_CONSTRUCTOR(a.x--,
                                a.y--,
                                a.z--,
                                a.w--);
}

occaFunction inline long4& operator -- (long4 &a){
  --a.x;
  --a.y;
  --a.z;
  --a.w;
  return a;
}
occaFunction inline long4  operator +  (const long4 &a, const long4 &b){
  return OCCA_LONG4_CONSTRUCTOR(a.x + b.x,
                                a.y + b.y,
                                a.z + b.z,
                                a.w + b.w);
}

occaFunction inline long4  operator +  (const long &a, const long4 &b){
  return OCCA_LONG4_CONSTRUCTOR(a + b.x,
                                a + b.y,
                                a + b.z,
                                a + b.w);
}

occaFunction inline long4  operator +  (const long4 &a, const long &b){
  return OCCA_LONG4_CONSTRUCTOR(a.x + b,
                                a.y + b,
                                a.z + b,
                                a.w + b);
}

occaFunction inline long4& operator += (      long4 &a, const long4 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

occaFunction inline long4& operator += (      long4 &a, const long &b){
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  return a;
}
occaFunction inline long4  operator -  (const long4 &a, const long4 &b){
  return OCCA_LONG4_CONSTRUCTOR(a.x - b.x,
                                a.y - b.y,
                                a.z - b.z,
                                a.w - b.w);
}

occaFunction inline long4  operator -  (const long &a, const long4 &b){
  return OCCA_LONG4_CONSTRUCTOR(a - b.x,
                                a - b.y,
                                a - b.z,
                                a - b.w);
}

occaFunction inline long4  operator -  (const long4 &a, const long &b){
  return OCCA_LONG4_CONSTRUCTOR(a.x - b,
                                a.y - b,
                                a.z - b,
                                a.w - b);
}

occaFunction inline long4& operator -= (      long4 &a, const long4 &b){
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  return a;
}

occaFunction inline long4& operator -= (      long4 &a, const long &b){
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  return a;
}
occaFunction inline long4  operator *  (const long4 &a, const long4 &b){
  return OCCA_LONG4_CONSTRUCTOR(a.x * b.x,
                                a.y * b.y,
                                a.z * b.z,
                                a.w * b.w);
}

occaFunction inline long4  operator *  (const long &a, const long4 &b){
  return OCCA_LONG4_CONSTRUCTOR(a * b.x,
                                a * b.y,
                                a * b.z,
                                a * b.w);
}

occaFunction inline long4  operator *  (const long4 &a, const long &b){
  return OCCA_LONG4_CONSTRUCTOR(a.x * b,
                                a.y * b,
                                a.z * b,
                                a.w * b);
}

occaFunction inline long4& operator *= (      long4 &a, const long4 &b){
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  return a;
}

occaFunction inline long4& operator *= (      long4 &a, const long &b){
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}
occaFunction inline long4  operator /  (const long4 &a, const long4 &b){
  return OCCA_LONG4_CONSTRUCTOR(a.x / b.x,
                                a.y / b.y,
                                a.z / b.z,
                                a.w / b.w);
}

occaFunction inline long4  operator /  (const long &a, const long4 &b){
  return OCCA_LONG4_CONSTRUCTOR(a / b.x,
                                a / b.y,
                                a / b.z,
                                a / b.w);
}

occaFunction inline long4  operator /  (const long4 &a, const long &b){
  return OCCA_LONG4_CONSTRUCTOR(a.x / b,
                                a.y / b,
                                a.z / b,
                                a.w / b);
}

occaFunction inline long4& operator /= (      long4 &a, const long4 &b){
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  return a;
}

occaFunction inline long4& operator /= (      long4 &a, const long &b){
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const long4& a){
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
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
typedef long4 long3;
#endif
//======================================


//---[ long8 ]--------------------------
#  define OCCA_LONG8_CONSTRUCTOR long8
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

occaFunction inline long8 operator + (const long8 &a){
  return OCCA_LONG8_CONSTRUCTOR(+a.x,
                                +a.y,
                                +a.z,
                                +a.w,
                                +a.s4,
                                +a.s5,
                                +a.s6,
                                +a.s7);
}

occaFunction inline long8 operator ++ (long8 &a, int){
  return OCCA_LONG8_CONSTRUCTOR(a.x++,
                                a.y++,
                                a.z++,
                                a.w++,
                                a.s4++,
                                a.s5++,
                                a.s6++,
                                a.s7++);
}

occaFunction inline long8& operator ++ (long8 &a){
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
occaFunction inline long8 operator - (const long8 &a){
  return OCCA_LONG8_CONSTRUCTOR(-a.x,
                                -a.y,
                                -a.z,
                                -a.w,
                                -a.s4,
                                -a.s5,
                                -a.s6,
                                -a.s7);
}

occaFunction inline long8 operator -- (long8 &a, int){
  return OCCA_LONG8_CONSTRUCTOR(a.x--,
                                a.y--,
                                a.z--,
                                a.w--,
                                a.s4--,
                                a.s5--,
                                a.s6--,
                                a.s7--);
}

occaFunction inline long8& operator -- (long8 &a){
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
occaFunction inline long8  operator +  (const long8 &a, const long8 &b){
  return OCCA_LONG8_CONSTRUCTOR(a.x + b.x,
                                a.y + b.y,
                                a.z + b.z,
                                a.w + b.w,
                                a.s4 + b.s4,
                                a.s5 + b.s5,
                                a.s6 + b.s6,
                                a.s7 + b.s7);
}

occaFunction inline long8  operator +  (const long &a, const long8 &b){
  return OCCA_LONG8_CONSTRUCTOR(a + b.x,
                                a + b.y,
                                a + b.z,
                                a + b.w,
                                a + b.s4,
                                a + b.s5,
                                a + b.s6,
                                a + b.s7);
}

occaFunction inline long8  operator +  (const long8 &a, const long &b){
  return OCCA_LONG8_CONSTRUCTOR(a.x + b,
                                a.y + b,
                                a.z + b,
                                a.w + b,
                                a.s4 + b,
                                a.s5 + b,
                                a.s6 + b,
                                a.s7 + b);
}

occaFunction inline long8& operator += (      long8 &a, const long8 &b){
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

occaFunction inline long8& operator += (      long8 &a, const long &b){
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
occaFunction inline long8  operator -  (const long8 &a, const long8 &b){
  return OCCA_LONG8_CONSTRUCTOR(a.x - b.x,
                                a.y - b.y,
                                a.z - b.z,
                                a.w - b.w,
                                a.s4 - b.s4,
                                a.s5 - b.s5,
                                a.s6 - b.s6,
                                a.s7 - b.s7);
}

occaFunction inline long8  operator -  (const long &a, const long8 &b){
  return OCCA_LONG8_CONSTRUCTOR(a - b.x,
                                a - b.y,
                                a - b.z,
                                a - b.w,
                                a - b.s4,
                                a - b.s5,
                                a - b.s6,
                                a - b.s7);
}

occaFunction inline long8  operator -  (const long8 &a, const long &b){
  return OCCA_LONG8_CONSTRUCTOR(a.x - b,
                                a.y - b,
                                a.z - b,
                                a.w - b,
                                a.s4 - b,
                                a.s5 - b,
                                a.s6 - b,
                                a.s7 - b);
}

occaFunction inline long8& operator -= (      long8 &a, const long8 &b){
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

occaFunction inline long8& operator -= (      long8 &a, const long &b){
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
occaFunction inline long8  operator *  (const long8 &a, const long8 &b){
  return OCCA_LONG8_CONSTRUCTOR(a.x * b.x,
                                a.y * b.y,
                                a.z * b.z,
                                a.w * b.w,
                                a.s4 * b.s4,
                                a.s5 * b.s5,
                                a.s6 * b.s6,
                                a.s7 * b.s7);
}

occaFunction inline long8  operator *  (const long &a, const long8 &b){
  return OCCA_LONG8_CONSTRUCTOR(a * b.x,
                                a * b.y,
                                a * b.z,
                                a * b.w,
                                a * b.s4,
                                a * b.s5,
                                a * b.s6,
                                a * b.s7);
}

occaFunction inline long8  operator *  (const long8 &a, const long &b){
  return OCCA_LONG8_CONSTRUCTOR(a.x * b,
                                a.y * b,
                                a.z * b,
                                a.w * b,
                                a.s4 * b,
                                a.s5 * b,
                                a.s6 * b,
                                a.s7 * b);
}

occaFunction inline long8& operator *= (      long8 &a, const long8 &b){
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

occaFunction inline long8& operator *= (      long8 &a, const long &b){
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
occaFunction inline long8  operator /  (const long8 &a, const long8 &b){
  return OCCA_LONG8_CONSTRUCTOR(a.x / b.x,
                                a.y / b.y,
                                a.z / b.z,
                                a.w / b.w,
                                a.s4 / b.s4,
                                a.s5 / b.s5,
                                a.s6 / b.s6,
                                a.s7 / b.s7);
}

occaFunction inline long8  operator /  (const long &a, const long8 &b){
  return OCCA_LONG8_CONSTRUCTOR(a / b.x,
                                a / b.y,
                                a / b.z,
                                a / b.w,
                                a / b.s4,
                                a / b.s5,
                                a / b.s6,
                                a / b.s7);
}

occaFunction inline long8  operator /  (const long8 &a, const long &b){
  return OCCA_LONG8_CONSTRUCTOR(a.x / b,
                                a.y / b,
                                a.z / b,
                                a.w / b,
                                a.s4 / b,
                                a.s5 / b,
                                a.s6 / b,
                                a.s7 / b);
}

occaFunction inline long8& operator /= (      long8 &a, const long8 &b){
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

occaFunction inline long8& operator /= (      long8 &a, const long &b){
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
inline std::ostream& operator << (std::ostream &out, const long8& a){
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
#  define OCCA_LONG16_CONSTRUCTOR long16
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

occaFunction inline long16 operator + (const long16 &a){
  return OCCA_LONG16_CONSTRUCTOR(+a.x,
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

occaFunction inline long16 operator ++ (long16 &a, int){
  return OCCA_LONG16_CONSTRUCTOR(a.x++,
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

occaFunction inline long16& operator ++ (long16 &a){
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
occaFunction inline long16 operator - (const long16 &a){
  return OCCA_LONG16_CONSTRUCTOR(-a.x,
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

occaFunction inline long16 operator -- (long16 &a, int){
  return OCCA_LONG16_CONSTRUCTOR(a.x--,
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

occaFunction inline long16& operator -- (long16 &a){
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
occaFunction inline long16  operator +  (const long16 &a, const long16 &b){
  return OCCA_LONG16_CONSTRUCTOR(a.x + b.x,
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

occaFunction inline long16  operator +  (const long &a, const long16 &b){
  return OCCA_LONG16_CONSTRUCTOR(a + b.x,
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

occaFunction inline long16  operator +  (const long16 &a, const long &b){
  return OCCA_LONG16_CONSTRUCTOR(a.x + b,
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

occaFunction inline long16& operator += (      long16 &a, const long16 &b){
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

occaFunction inline long16& operator += (      long16 &a, const long &b){
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
occaFunction inline long16  operator -  (const long16 &a, const long16 &b){
  return OCCA_LONG16_CONSTRUCTOR(a.x - b.x,
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

occaFunction inline long16  operator -  (const long &a, const long16 &b){
  return OCCA_LONG16_CONSTRUCTOR(a - b.x,
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

occaFunction inline long16  operator -  (const long16 &a, const long &b){
  return OCCA_LONG16_CONSTRUCTOR(a.x - b,
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

occaFunction inline long16& operator -= (      long16 &a, const long16 &b){
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

occaFunction inline long16& operator -= (      long16 &a, const long &b){
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
occaFunction inline long16  operator *  (const long16 &a, const long16 &b){
  return OCCA_LONG16_CONSTRUCTOR(a.x * b.x,
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

occaFunction inline long16  operator *  (const long &a, const long16 &b){
  return OCCA_LONG16_CONSTRUCTOR(a * b.x,
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

occaFunction inline long16  operator *  (const long16 &a, const long &b){
  return OCCA_LONG16_CONSTRUCTOR(a.x * b,
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

occaFunction inline long16& operator *= (      long16 &a, const long16 &b){
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

occaFunction inline long16& operator *= (      long16 &a, const long &b){
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
occaFunction inline long16  operator /  (const long16 &a, const long16 &b){
  return OCCA_LONG16_CONSTRUCTOR(a.x / b.x,
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

occaFunction inline long16  operator /  (const long &a, const long16 &b){
  return OCCA_LONG16_CONSTRUCTOR(a / b.x,
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

occaFunction inline long16  operator /  (const long16 &a, const long &b){
  return OCCA_LONG16_CONSTRUCTOR(a.x / b,
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

occaFunction inline long16& operator /= (      long16 &a, const long16 &b){
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

occaFunction inline long16& operator /= (      long16 &a, const long &b){
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
inline std::ostream& operator << (std::ostream &out, const long16& a){
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
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
#  define OCCA_FLOAT2_CONSTRUCTOR float2
#else
#  define OCCA_FLOAT2_CONSTRUCTOR make_float2
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

occaFunction inline float2 operator + (const float2 &a){
  return OCCA_FLOAT2_CONSTRUCTOR(+a.x,
                                 +a.y);
}
occaFunction inline float2 operator - (const float2 &a){
  return OCCA_FLOAT2_CONSTRUCTOR(-a.x,
                                 -a.y);
}
occaFunction inline float2  operator +  (const float2 &a, const float2 &b){
  return OCCA_FLOAT2_CONSTRUCTOR(a.x + b.x,
                                 a.y + b.y);
}

occaFunction inline float2  operator +  (const float &a, const float2 &b){
  return OCCA_FLOAT2_CONSTRUCTOR(a + b.x,
                                 a + b.y);
}

occaFunction inline float2  operator +  (const float2 &a, const float &b){
  return OCCA_FLOAT2_CONSTRUCTOR(a.x + b,
                                 a.y + b);
}

occaFunction inline float2& operator += (      float2 &a, const float2 &b){
  a.x += b.x;
  a.y += b.y;
  return a;
}

occaFunction inline float2& operator += (      float2 &a, const float &b){
  a.x += b;
  a.y += b;
  return a;
}
occaFunction inline float2  operator -  (const float2 &a, const float2 &b){
  return OCCA_FLOAT2_CONSTRUCTOR(a.x - b.x,
                                 a.y - b.y);
}

occaFunction inline float2  operator -  (const float &a, const float2 &b){
  return OCCA_FLOAT2_CONSTRUCTOR(a - b.x,
                                 a - b.y);
}

occaFunction inline float2  operator -  (const float2 &a, const float &b){
  return OCCA_FLOAT2_CONSTRUCTOR(a.x - b,
                                 a.y - b);
}

occaFunction inline float2& operator -= (      float2 &a, const float2 &b){
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

occaFunction inline float2& operator -= (      float2 &a, const float &b){
  a.x -= b;
  a.y -= b;
  return a;
}
occaFunction inline float2  operator *  (const float2 &a, const float2 &b){
  return OCCA_FLOAT2_CONSTRUCTOR(a.x * b.x,
                                 a.y * b.y);
}

occaFunction inline float2  operator *  (const float &a, const float2 &b){
  return OCCA_FLOAT2_CONSTRUCTOR(a * b.x,
                                 a * b.y);
}

occaFunction inline float2  operator *  (const float2 &a, const float &b){
  return OCCA_FLOAT2_CONSTRUCTOR(a.x * b,
                                 a.y * b);
}

occaFunction inline float2& operator *= (      float2 &a, const float2 &b){
  a.x *= b.x;
  a.y *= b.y;
  return a;
}

occaFunction inline float2& operator *= (      float2 &a, const float &b){
  a.x *= b;
  a.y *= b;
  return a;
}
occaFunction inline float2  operator /  (const float2 &a, const float2 &b){
  return OCCA_FLOAT2_CONSTRUCTOR(a.x / b.x,
                                 a.y / b.y);
}

occaFunction inline float2  operator /  (const float &a, const float2 &b){
  return OCCA_FLOAT2_CONSTRUCTOR(a / b.x,
                                 a / b.y);
}

occaFunction inline float2  operator /  (const float2 &a, const float &b){
  return OCCA_FLOAT2_CONSTRUCTOR(a.x / b,
                                 a.y / b);
}

occaFunction inline float2& operator /= (      float2 &a, const float2 &b){
  a.x /= b.x;
  a.y /= b.y;
  return a;
}

occaFunction inline float2& operator /= (      float2 &a, const float &b){
  a.x /= b;
  a.y /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const float2& a){
  out << "[" << a.x << ", "
             << a.y
      << "]\n";

  return out;
}
#endif

//======================================


//---[ float4 ]-------------------------
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
#  define OCCA_FLOAT4_CONSTRUCTOR float4
#else
#  define OCCA_FLOAT4_CONSTRUCTOR make_float4
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

occaFunction inline float4 operator + (const float4 &a){
  return OCCA_FLOAT4_CONSTRUCTOR(+a.x,
                                 +a.y,
                                 +a.z,
                                 +a.w);
}
occaFunction inline float4 operator - (const float4 &a){
  return OCCA_FLOAT4_CONSTRUCTOR(-a.x,
                                 -a.y,
                                 -a.z,
                                 -a.w);
}
occaFunction inline float4  operator +  (const float4 &a, const float4 &b){
  return OCCA_FLOAT4_CONSTRUCTOR(a.x + b.x,
                                 a.y + b.y,
                                 a.z + b.z,
                                 a.w + b.w);
}

occaFunction inline float4  operator +  (const float &a, const float4 &b){
  return OCCA_FLOAT4_CONSTRUCTOR(a + b.x,
                                 a + b.y,
                                 a + b.z,
                                 a + b.w);
}

occaFunction inline float4  operator +  (const float4 &a, const float &b){
  return OCCA_FLOAT4_CONSTRUCTOR(a.x + b,
                                 a.y + b,
                                 a.z + b,
                                 a.w + b);
}

occaFunction inline float4& operator += (      float4 &a, const float4 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

occaFunction inline float4& operator += (      float4 &a, const float &b){
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  return a;
}
occaFunction inline float4  operator -  (const float4 &a, const float4 &b){
  return OCCA_FLOAT4_CONSTRUCTOR(a.x - b.x,
                                 a.y - b.y,
                                 a.z - b.z,
                                 a.w - b.w);
}

occaFunction inline float4  operator -  (const float &a, const float4 &b){
  return OCCA_FLOAT4_CONSTRUCTOR(a - b.x,
                                 a - b.y,
                                 a - b.z,
                                 a - b.w);
}

occaFunction inline float4  operator -  (const float4 &a, const float &b){
  return OCCA_FLOAT4_CONSTRUCTOR(a.x - b,
                                 a.y - b,
                                 a.z - b,
                                 a.w - b);
}

occaFunction inline float4& operator -= (      float4 &a, const float4 &b){
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  return a;
}

occaFunction inline float4& operator -= (      float4 &a, const float &b){
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  return a;
}
occaFunction inline float4  operator *  (const float4 &a, const float4 &b){
  return OCCA_FLOAT4_CONSTRUCTOR(a.x * b.x,
                                 a.y * b.y,
                                 a.z * b.z,
                                 a.w * b.w);
}

occaFunction inline float4  operator *  (const float &a, const float4 &b){
  return OCCA_FLOAT4_CONSTRUCTOR(a * b.x,
                                 a * b.y,
                                 a * b.z,
                                 a * b.w);
}

occaFunction inline float4  operator *  (const float4 &a, const float &b){
  return OCCA_FLOAT4_CONSTRUCTOR(a.x * b,
                                 a.y * b,
                                 a.z * b,
                                 a.w * b);
}

occaFunction inline float4& operator *= (      float4 &a, const float4 &b){
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  return a;
}

occaFunction inline float4& operator *= (      float4 &a, const float &b){
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}
occaFunction inline float4  operator /  (const float4 &a, const float4 &b){
  return OCCA_FLOAT4_CONSTRUCTOR(a.x / b.x,
                                 a.y / b.y,
                                 a.z / b.z,
                                 a.w / b.w);
}

occaFunction inline float4  operator /  (const float &a, const float4 &b){
  return OCCA_FLOAT4_CONSTRUCTOR(a / b.x,
                                 a / b.y,
                                 a / b.z,
                                 a / b.w);
}

occaFunction inline float4  operator /  (const float4 &a, const float &b){
  return OCCA_FLOAT4_CONSTRUCTOR(a.x / b,
                                 a.y / b,
                                 a.z / b,
                                 a.w / b);
}

occaFunction inline float4& operator /= (      float4 &a, const float4 &b){
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  return a;
}

occaFunction inline float4& operator /= (      float4 &a, const float &b){
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const float4& a){
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
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
typedef float4 float3;
#endif
//======================================


//---[ float8 ]-------------------------
#  define OCCA_FLOAT8_CONSTRUCTOR float8
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

occaFunction inline float8 operator + (const float8 &a){
  return OCCA_FLOAT8_CONSTRUCTOR(+a.x,
                                 +a.y,
                                 +a.z,
                                 +a.w,
                                 +a.s4,
                                 +a.s5,
                                 +a.s6,
                                 +a.s7);
}
occaFunction inline float8 operator - (const float8 &a){
  return OCCA_FLOAT8_CONSTRUCTOR(-a.x,
                                 -a.y,
                                 -a.z,
                                 -a.w,
                                 -a.s4,
                                 -a.s5,
                                 -a.s6,
                                 -a.s7);
}
occaFunction inline float8  operator +  (const float8 &a, const float8 &b){
  return OCCA_FLOAT8_CONSTRUCTOR(a.x + b.x,
                                 a.y + b.y,
                                 a.z + b.z,
                                 a.w + b.w,
                                 a.s4 + b.s4,
                                 a.s5 + b.s5,
                                 a.s6 + b.s6,
                                 a.s7 + b.s7);
}

occaFunction inline float8  operator +  (const float &a, const float8 &b){
  return OCCA_FLOAT8_CONSTRUCTOR(a + b.x,
                                 a + b.y,
                                 a + b.z,
                                 a + b.w,
                                 a + b.s4,
                                 a + b.s5,
                                 a + b.s6,
                                 a + b.s7);
}

occaFunction inline float8  operator +  (const float8 &a, const float &b){
  return OCCA_FLOAT8_CONSTRUCTOR(a.x + b,
                                 a.y + b,
                                 a.z + b,
                                 a.w + b,
                                 a.s4 + b,
                                 a.s5 + b,
                                 a.s6 + b,
                                 a.s7 + b);
}

occaFunction inline float8& operator += (      float8 &a, const float8 &b){
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

occaFunction inline float8& operator += (      float8 &a, const float &b){
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
occaFunction inline float8  operator -  (const float8 &a, const float8 &b){
  return OCCA_FLOAT8_CONSTRUCTOR(a.x - b.x,
                                 a.y - b.y,
                                 a.z - b.z,
                                 a.w - b.w,
                                 a.s4 - b.s4,
                                 a.s5 - b.s5,
                                 a.s6 - b.s6,
                                 a.s7 - b.s7);
}

occaFunction inline float8  operator -  (const float &a, const float8 &b){
  return OCCA_FLOAT8_CONSTRUCTOR(a - b.x,
                                 a - b.y,
                                 a - b.z,
                                 a - b.w,
                                 a - b.s4,
                                 a - b.s5,
                                 a - b.s6,
                                 a - b.s7);
}

occaFunction inline float8  operator -  (const float8 &a, const float &b){
  return OCCA_FLOAT8_CONSTRUCTOR(a.x - b,
                                 a.y - b,
                                 a.z - b,
                                 a.w - b,
                                 a.s4 - b,
                                 a.s5 - b,
                                 a.s6 - b,
                                 a.s7 - b);
}

occaFunction inline float8& operator -= (      float8 &a, const float8 &b){
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

occaFunction inline float8& operator -= (      float8 &a, const float &b){
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
occaFunction inline float8  operator *  (const float8 &a, const float8 &b){
  return OCCA_FLOAT8_CONSTRUCTOR(a.x * b.x,
                                 a.y * b.y,
                                 a.z * b.z,
                                 a.w * b.w,
                                 a.s4 * b.s4,
                                 a.s5 * b.s5,
                                 a.s6 * b.s6,
                                 a.s7 * b.s7);
}

occaFunction inline float8  operator *  (const float &a, const float8 &b){
  return OCCA_FLOAT8_CONSTRUCTOR(a * b.x,
                                 a * b.y,
                                 a * b.z,
                                 a * b.w,
                                 a * b.s4,
                                 a * b.s5,
                                 a * b.s6,
                                 a * b.s7);
}

occaFunction inline float8  operator *  (const float8 &a, const float &b){
  return OCCA_FLOAT8_CONSTRUCTOR(a.x * b,
                                 a.y * b,
                                 a.z * b,
                                 a.w * b,
                                 a.s4 * b,
                                 a.s5 * b,
                                 a.s6 * b,
                                 a.s7 * b);
}

occaFunction inline float8& operator *= (      float8 &a, const float8 &b){
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

occaFunction inline float8& operator *= (      float8 &a, const float &b){
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
occaFunction inline float8  operator /  (const float8 &a, const float8 &b){
  return OCCA_FLOAT8_CONSTRUCTOR(a.x / b.x,
                                 a.y / b.y,
                                 a.z / b.z,
                                 a.w / b.w,
                                 a.s4 / b.s4,
                                 a.s5 / b.s5,
                                 a.s6 / b.s6,
                                 a.s7 / b.s7);
}

occaFunction inline float8  operator /  (const float &a, const float8 &b){
  return OCCA_FLOAT8_CONSTRUCTOR(a / b.x,
                                 a / b.y,
                                 a / b.z,
                                 a / b.w,
                                 a / b.s4,
                                 a / b.s5,
                                 a / b.s6,
                                 a / b.s7);
}

occaFunction inline float8  operator /  (const float8 &a, const float &b){
  return OCCA_FLOAT8_CONSTRUCTOR(a.x / b,
                                 a.y / b,
                                 a.z / b,
                                 a.w / b,
                                 a.s4 / b,
                                 a.s5 / b,
                                 a.s6 / b,
                                 a.s7 / b);
}

occaFunction inline float8& operator /= (      float8 &a, const float8 &b){
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

occaFunction inline float8& operator /= (      float8 &a, const float &b){
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
inline std::ostream& operator << (std::ostream &out, const float8& a){
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
#  define OCCA_FLOAT16_CONSTRUCTOR float16
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

occaFunction inline float16 operator + (const float16 &a){
  return OCCA_FLOAT16_CONSTRUCTOR(+a.x,
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
occaFunction inline float16 operator - (const float16 &a){
  return OCCA_FLOAT16_CONSTRUCTOR(-a.x,
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
occaFunction inline float16  operator +  (const float16 &a, const float16 &b){
  return OCCA_FLOAT16_CONSTRUCTOR(a.x + b.x,
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

occaFunction inline float16  operator +  (const float &a, const float16 &b){
  return OCCA_FLOAT16_CONSTRUCTOR(a + b.x,
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

occaFunction inline float16  operator +  (const float16 &a, const float &b){
  return OCCA_FLOAT16_CONSTRUCTOR(a.x + b,
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

occaFunction inline float16& operator += (      float16 &a, const float16 &b){
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

occaFunction inline float16& operator += (      float16 &a, const float &b){
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
occaFunction inline float16  operator -  (const float16 &a, const float16 &b){
  return OCCA_FLOAT16_CONSTRUCTOR(a.x - b.x,
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

occaFunction inline float16  operator -  (const float &a, const float16 &b){
  return OCCA_FLOAT16_CONSTRUCTOR(a - b.x,
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

occaFunction inline float16  operator -  (const float16 &a, const float &b){
  return OCCA_FLOAT16_CONSTRUCTOR(a.x - b,
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

occaFunction inline float16& operator -= (      float16 &a, const float16 &b){
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

occaFunction inline float16& operator -= (      float16 &a, const float &b){
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
occaFunction inline float16  operator *  (const float16 &a, const float16 &b){
  return OCCA_FLOAT16_CONSTRUCTOR(a.x * b.x,
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

occaFunction inline float16  operator *  (const float &a, const float16 &b){
  return OCCA_FLOAT16_CONSTRUCTOR(a * b.x,
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

occaFunction inline float16  operator *  (const float16 &a, const float &b){
  return OCCA_FLOAT16_CONSTRUCTOR(a.x * b,
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

occaFunction inline float16& operator *= (      float16 &a, const float16 &b){
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

occaFunction inline float16& operator *= (      float16 &a, const float &b){
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
occaFunction inline float16  operator /  (const float16 &a, const float16 &b){
  return OCCA_FLOAT16_CONSTRUCTOR(a.x / b.x,
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

occaFunction inline float16  operator /  (const float &a, const float16 &b){
  return OCCA_FLOAT16_CONSTRUCTOR(a / b.x,
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

occaFunction inline float16  operator /  (const float16 &a, const float &b){
  return OCCA_FLOAT16_CONSTRUCTOR(a.x / b,
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

occaFunction inline float16& operator /= (      float16 &a, const float16 &b){
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

occaFunction inline float16& operator /= (      float16 &a, const float &b){
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
inline std::ostream& operator << (std::ostream &out, const float16& a){
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
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
#  define OCCA_DOUBLE2_CONSTRUCTOR double2
#else
#  define OCCA_DOUBLE2_CONSTRUCTOR make_double2
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

occaFunction inline double2 operator + (const double2 &a){
  return OCCA_DOUBLE2_CONSTRUCTOR(+a.x,
                                  +a.y);
}
occaFunction inline double2 operator - (const double2 &a){
  return OCCA_DOUBLE2_CONSTRUCTOR(-a.x,
                                  -a.y);
}
occaFunction inline double2  operator +  (const double2 &a, const double2 &b){
  return OCCA_DOUBLE2_CONSTRUCTOR(a.x + b.x,
                                  a.y + b.y);
}

occaFunction inline double2  operator +  (const double &a, const double2 &b){
  return OCCA_DOUBLE2_CONSTRUCTOR(a + b.x,
                                  a + b.y);
}

occaFunction inline double2  operator +  (const double2 &a, const double &b){
  return OCCA_DOUBLE2_CONSTRUCTOR(a.x + b,
                                  a.y + b);
}

occaFunction inline double2& operator += (      double2 &a, const double2 &b){
  a.x += b.x;
  a.y += b.y;
  return a;
}

occaFunction inline double2& operator += (      double2 &a, const double &b){
  a.x += b;
  a.y += b;
  return a;
}
occaFunction inline double2  operator -  (const double2 &a, const double2 &b){
  return OCCA_DOUBLE2_CONSTRUCTOR(a.x - b.x,
                                  a.y - b.y);
}

occaFunction inline double2  operator -  (const double &a, const double2 &b){
  return OCCA_DOUBLE2_CONSTRUCTOR(a - b.x,
                                  a - b.y);
}

occaFunction inline double2  operator -  (const double2 &a, const double &b){
  return OCCA_DOUBLE2_CONSTRUCTOR(a.x - b,
                                  a.y - b);
}

occaFunction inline double2& operator -= (      double2 &a, const double2 &b){
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

occaFunction inline double2& operator -= (      double2 &a, const double &b){
  a.x -= b;
  a.y -= b;
  return a;
}
occaFunction inline double2  operator *  (const double2 &a, const double2 &b){
  return OCCA_DOUBLE2_CONSTRUCTOR(a.x * b.x,
                                  a.y * b.y);
}

occaFunction inline double2  operator *  (const double &a, const double2 &b){
  return OCCA_DOUBLE2_CONSTRUCTOR(a * b.x,
                                  a * b.y);
}

occaFunction inline double2  operator *  (const double2 &a, const double &b){
  return OCCA_DOUBLE2_CONSTRUCTOR(a.x * b,
                                  a.y * b);
}

occaFunction inline double2& operator *= (      double2 &a, const double2 &b){
  a.x *= b.x;
  a.y *= b.y;
  return a;
}

occaFunction inline double2& operator *= (      double2 &a, const double &b){
  a.x *= b;
  a.y *= b;
  return a;
}
occaFunction inline double2  operator /  (const double2 &a, const double2 &b){
  return OCCA_DOUBLE2_CONSTRUCTOR(a.x / b.x,
                                  a.y / b.y);
}

occaFunction inline double2  operator /  (const double &a, const double2 &b){
  return OCCA_DOUBLE2_CONSTRUCTOR(a / b.x,
                                  a / b.y);
}

occaFunction inline double2  operator /  (const double2 &a, const double &b){
  return OCCA_DOUBLE2_CONSTRUCTOR(a.x / b,
                                  a.y / b);
}

occaFunction inline double2& operator /= (      double2 &a, const double2 &b){
  a.x /= b.x;
  a.y /= b.y;
  return a;
}

occaFunction inline double2& operator /= (      double2 &a, const double &b){
  a.x /= b;
  a.y /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const double2& a){
  out << "[" << a.x << ", "
             << a.y
      << "]\n";

  return out;
}
#endif

//======================================


//---[ double4 ]------------------------
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
#  define OCCA_DOUBLE4_CONSTRUCTOR double4
#else
#  define OCCA_DOUBLE4_CONSTRUCTOR make_double4
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

occaFunction inline double4 operator + (const double4 &a){
  return OCCA_DOUBLE4_CONSTRUCTOR(+a.x,
                                  +a.y,
                                  +a.z,
                                  +a.w);
}
occaFunction inline double4 operator - (const double4 &a){
  return OCCA_DOUBLE4_CONSTRUCTOR(-a.x,
                                  -a.y,
                                  -a.z,
                                  -a.w);
}
occaFunction inline double4  operator +  (const double4 &a, const double4 &b){
  return OCCA_DOUBLE4_CONSTRUCTOR(a.x + b.x,
                                  a.y + b.y,
                                  a.z + b.z,
                                  a.w + b.w);
}

occaFunction inline double4  operator +  (const double &a, const double4 &b){
  return OCCA_DOUBLE4_CONSTRUCTOR(a + b.x,
                                  a + b.y,
                                  a + b.z,
                                  a + b.w);
}

occaFunction inline double4  operator +  (const double4 &a, const double &b){
  return OCCA_DOUBLE4_CONSTRUCTOR(a.x + b,
                                  a.y + b,
                                  a.z + b,
                                  a.w + b);
}

occaFunction inline double4& operator += (      double4 &a, const double4 &b){
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

occaFunction inline double4& operator += (      double4 &a, const double &b){
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
  return a;
}
occaFunction inline double4  operator -  (const double4 &a, const double4 &b){
  return OCCA_DOUBLE4_CONSTRUCTOR(a.x - b.x,
                                  a.y - b.y,
                                  a.z - b.z,
                                  a.w - b.w);
}

occaFunction inline double4  operator -  (const double &a, const double4 &b){
  return OCCA_DOUBLE4_CONSTRUCTOR(a - b.x,
                                  a - b.y,
                                  a - b.z,
                                  a - b.w);
}

occaFunction inline double4  operator -  (const double4 &a, const double &b){
  return OCCA_DOUBLE4_CONSTRUCTOR(a.x - b,
                                  a.y - b,
                                  a.z - b,
                                  a.w - b);
}

occaFunction inline double4& operator -= (      double4 &a, const double4 &b){
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
  return a;
}

occaFunction inline double4& operator -= (      double4 &a, const double &b){
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
  return a;
}
occaFunction inline double4  operator *  (const double4 &a, const double4 &b){
  return OCCA_DOUBLE4_CONSTRUCTOR(a.x * b.x,
                                  a.y * b.y,
                                  a.z * b.z,
                                  a.w * b.w);
}

occaFunction inline double4  operator *  (const double &a, const double4 &b){
  return OCCA_DOUBLE4_CONSTRUCTOR(a * b.x,
                                  a * b.y,
                                  a * b.z,
                                  a * b.w);
}

occaFunction inline double4  operator *  (const double4 &a, const double &b){
  return OCCA_DOUBLE4_CONSTRUCTOR(a.x * b,
                                  a.y * b,
                                  a.z * b,
                                  a.w * b);
}

occaFunction inline double4& operator *= (      double4 &a, const double4 &b){
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
  return a;
}

occaFunction inline double4& operator *= (      double4 &a, const double &b){
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}
occaFunction inline double4  operator /  (const double4 &a, const double4 &b){
  return OCCA_DOUBLE4_CONSTRUCTOR(a.x / b.x,
                                  a.y / b.y,
                                  a.z / b.z,
                                  a.w / b.w);
}

occaFunction inline double4  operator /  (const double &a, const double4 &b){
  return OCCA_DOUBLE4_CONSTRUCTOR(a / b.x,
                                  a / b.y,
                                  a / b.z,
                                  a / b.w);
}

occaFunction inline double4  operator /  (const double4 &a, const double &b){
  return OCCA_DOUBLE4_CONSTRUCTOR(a.x / b,
                                  a.y / b,
                                  a.z / b,
                                  a.w / b);
}

occaFunction inline double4& operator /= (      double4 &a, const double4 &b){
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
  return a;
}

occaFunction inline double4& operator /= (      double4 &a, const double &b){
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
  return a;
}

#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS))
inline std::ostream& operator << (std::ostream &out, const double4& a){
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
#if (!defined(OCCA_IN_KERNEL) || (OCCA_USING_CUDA == 0))
typedef double4 double3;
#endif
//======================================


//---[ double8 ]------------------------
#  define OCCA_DOUBLE8_CONSTRUCTOR double8
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

occaFunction inline double8 operator + (const double8 &a){
  return OCCA_DOUBLE8_CONSTRUCTOR(+a.x,
                                  +a.y,
                                  +a.z,
                                  +a.w,
                                  +a.s4,
                                  +a.s5,
                                  +a.s6,
                                  +a.s7);
}
occaFunction inline double8 operator - (const double8 &a){
  return OCCA_DOUBLE8_CONSTRUCTOR(-a.x,
                                  -a.y,
                                  -a.z,
                                  -a.w,
                                  -a.s4,
                                  -a.s5,
                                  -a.s6,
                                  -a.s7);
}
occaFunction inline double8  operator +  (const double8 &a, const double8 &b){
  return OCCA_DOUBLE8_CONSTRUCTOR(a.x + b.x,
                                  a.y + b.y,
                                  a.z + b.z,
                                  a.w + b.w,
                                  a.s4 + b.s4,
                                  a.s5 + b.s5,
                                  a.s6 + b.s6,
                                  a.s7 + b.s7);
}

occaFunction inline double8  operator +  (const double &a, const double8 &b){
  return OCCA_DOUBLE8_CONSTRUCTOR(a + b.x,
                                  a + b.y,
                                  a + b.z,
                                  a + b.w,
                                  a + b.s4,
                                  a + b.s5,
                                  a + b.s6,
                                  a + b.s7);
}

occaFunction inline double8  operator +  (const double8 &a, const double &b){
  return OCCA_DOUBLE8_CONSTRUCTOR(a.x + b,
                                  a.y + b,
                                  a.z + b,
                                  a.w + b,
                                  a.s4 + b,
                                  a.s5 + b,
                                  a.s6 + b,
                                  a.s7 + b);
}

occaFunction inline double8& operator += (      double8 &a, const double8 &b){
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

occaFunction inline double8& operator += (      double8 &a, const double &b){
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
occaFunction inline double8  operator -  (const double8 &a, const double8 &b){
  return OCCA_DOUBLE8_CONSTRUCTOR(a.x - b.x,
                                  a.y - b.y,
                                  a.z - b.z,
                                  a.w - b.w,
                                  a.s4 - b.s4,
                                  a.s5 - b.s5,
                                  a.s6 - b.s6,
                                  a.s7 - b.s7);
}

occaFunction inline double8  operator -  (const double &a, const double8 &b){
  return OCCA_DOUBLE8_CONSTRUCTOR(a - b.x,
                                  a - b.y,
                                  a - b.z,
                                  a - b.w,
                                  a - b.s4,
                                  a - b.s5,
                                  a - b.s6,
                                  a - b.s7);
}

occaFunction inline double8  operator -  (const double8 &a, const double &b){
  return OCCA_DOUBLE8_CONSTRUCTOR(a.x - b,
                                  a.y - b,
                                  a.z - b,
                                  a.w - b,
                                  a.s4 - b,
                                  a.s5 - b,
                                  a.s6 - b,
                                  a.s7 - b);
}

occaFunction inline double8& operator -= (      double8 &a, const double8 &b){
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

occaFunction inline double8& operator -= (      double8 &a, const double &b){
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
occaFunction inline double8  operator *  (const double8 &a, const double8 &b){
  return OCCA_DOUBLE8_CONSTRUCTOR(a.x * b.x,
                                  a.y * b.y,
                                  a.z * b.z,
                                  a.w * b.w,
                                  a.s4 * b.s4,
                                  a.s5 * b.s5,
                                  a.s6 * b.s6,
                                  a.s7 * b.s7);
}

occaFunction inline double8  operator *  (const double &a, const double8 &b){
  return OCCA_DOUBLE8_CONSTRUCTOR(a * b.x,
                                  a * b.y,
                                  a * b.z,
                                  a * b.w,
                                  a * b.s4,
                                  a * b.s5,
                                  a * b.s6,
                                  a * b.s7);
}

occaFunction inline double8  operator *  (const double8 &a, const double &b){
  return OCCA_DOUBLE8_CONSTRUCTOR(a.x * b,
                                  a.y * b,
                                  a.z * b,
                                  a.w * b,
                                  a.s4 * b,
                                  a.s5 * b,
                                  a.s6 * b,
                                  a.s7 * b);
}

occaFunction inline double8& operator *= (      double8 &a, const double8 &b){
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

occaFunction inline double8& operator *= (      double8 &a, const double &b){
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
occaFunction inline double8  operator /  (const double8 &a, const double8 &b){
  return OCCA_DOUBLE8_CONSTRUCTOR(a.x / b.x,
                                  a.y / b.y,
                                  a.z / b.z,
                                  a.w / b.w,
                                  a.s4 / b.s4,
                                  a.s5 / b.s5,
                                  a.s6 / b.s6,
                                  a.s7 / b.s7);
}

occaFunction inline double8  operator /  (const double &a, const double8 &b){
  return OCCA_DOUBLE8_CONSTRUCTOR(a / b.x,
                                  a / b.y,
                                  a / b.z,
                                  a / b.w,
                                  a / b.s4,
                                  a / b.s5,
                                  a / b.s6,
                                  a / b.s7);
}

occaFunction inline double8  operator /  (const double8 &a, const double &b){
  return OCCA_DOUBLE8_CONSTRUCTOR(a.x / b,
                                  a.y / b,
                                  a.z / b,
                                  a.w / b,
                                  a.s4 / b,
                                  a.s5 / b,
                                  a.s6 / b,
                                  a.s7 / b);
}

occaFunction inline double8& operator /= (      double8 &a, const double8 &b){
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

occaFunction inline double8& operator /= (      double8 &a, const double &b){
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
inline std::ostream& operator << (std::ostream &out, const double8& a){
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
#  define OCCA_DOUBLE16_CONSTRUCTOR double16
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

occaFunction inline double16 operator + (const double16 &a){
  return OCCA_DOUBLE16_CONSTRUCTOR(+a.x,
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
occaFunction inline double16 operator - (const double16 &a){
  return OCCA_DOUBLE16_CONSTRUCTOR(-a.x,
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
occaFunction inline double16  operator +  (const double16 &a, const double16 &b){
  return OCCA_DOUBLE16_CONSTRUCTOR(a.x + b.x,
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

occaFunction inline double16  operator +  (const double &a, const double16 &b){
  return OCCA_DOUBLE16_CONSTRUCTOR(a + b.x,
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

occaFunction inline double16  operator +  (const double16 &a, const double &b){
  return OCCA_DOUBLE16_CONSTRUCTOR(a.x + b,
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

occaFunction inline double16& operator += (      double16 &a, const double16 &b){
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

occaFunction inline double16& operator += (      double16 &a, const double &b){
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
occaFunction inline double16  operator -  (const double16 &a, const double16 &b){
  return OCCA_DOUBLE16_CONSTRUCTOR(a.x - b.x,
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

occaFunction inline double16  operator -  (const double &a, const double16 &b){
  return OCCA_DOUBLE16_CONSTRUCTOR(a - b.x,
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

occaFunction inline double16  operator -  (const double16 &a, const double &b){
  return OCCA_DOUBLE16_CONSTRUCTOR(a.x - b,
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

occaFunction inline double16& operator -= (      double16 &a, const double16 &b){
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

occaFunction inline double16& operator -= (      double16 &a, const double &b){
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
occaFunction inline double16  operator *  (const double16 &a, const double16 &b){
  return OCCA_DOUBLE16_CONSTRUCTOR(a.x * b.x,
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

occaFunction inline double16  operator *  (const double &a, const double16 &b){
  return OCCA_DOUBLE16_CONSTRUCTOR(a * b.x,
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

occaFunction inline double16  operator *  (const double16 &a, const double &b){
  return OCCA_DOUBLE16_CONSTRUCTOR(a.x * b,
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

occaFunction inline double16& operator *= (      double16 &a, const double16 &b){
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

occaFunction inline double16& operator *= (      double16 &a, const double &b){
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
occaFunction inline double16  operator /  (const double16 &a, const double16 &b){
  return OCCA_DOUBLE16_CONSTRUCTOR(a.x / b.x,
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

occaFunction inline double16  operator /  (const double &a, const double16 &b){
  return OCCA_DOUBLE16_CONSTRUCTOR(a / b.x,
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

occaFunction inline double16  operator /  (const double16 &a, const double &b){
  return OCCA_DOUBLE16_CONSTRUCTOR(a.x / b,
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

occaFunction inline double16& operator /= (      double16 &a, const double16 &b){
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

occaFunction inline double16& operator /= (      double16 &a, const double &b){
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
inline std::ostream& operator << (std::ostream &out, const double16& a){
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


#if OCCA_USING_CPU && (OCCA_COMPILED_WITH & OCCA_INTEL_COMPILER)
#  define OCCA_CPU_SIMD_WIDTH OCCA_SIMD_WIDTH
#else
#  define OCCA_CPU_SIMD_WIDTH 0
#endif

#if 4 <= OCCA_CPU_SIMD_WIDTH
#  define occaLoadF4(DEST, SRC)   *((_m128*)&DEST) = __mm_load_ps((float*)&SRC)
#  define occaStoreF4(DEST, SRC)  _mm_store_ps((float*)&DEST, *((_m128*)&SRC)
#  define occaAddF4(V12, V1, V2)  *((_m128*)&V12) = __mm_add_ps(*((_m128*)&V1), *((_m128*)&V2))
#  define occaMultF4(V12, V1, V2) *((_m128*)&V12) = __mm_mul_ps(*((_m128*)&V1), *((_m128*)&V2))
#else
#  define occaLoadF4(DEST, SRC)   DEST = SRC
#  define occaStoreF4(DEST, SRC)  DEST = SRC
#  define occaAddF4(V12, V1, V2)  V12 = (V1 + V2)
#  define occaMultF4(V12, V1, V2) V12 = (V1 * V2)
#endif

#if 8 <= OCCA_CPU_SIMD_WIDTH
#  define occaLoadF8(DEST, SRC)   *((_m256*)&DEST) = __mm256_load_ps((float*)&SRC)
#  define occaStoreF8(DEST, SRC)  _mm256_store_ps((float*)&DEST, *((_m256*)&SRC)
#  define occaAddF8(V12, V1, V2)  *((_m256*)&V12) = __mm256_add_ps(*((_m256*)&V1), *((_m256*)&V2))
#  define occaMultF8(V12, V1, V2) *((_m256*)&V12) = __mm256_mul_ps(*((_m256*)&V1), *((_m256*)&V2))
#else
#  define occaLoadF8(DEST, SRC)   DEST = SRC
#  define occaStoreF8(DEST, SRC)  DEST = SRC
#  define occaAddF8(V12, V1, V2)  V12 = (V1 + V2)
#  define occaMultF8(V12, V1, V2) V12 = (V1 * V2)
#endif


struct vfloat2 {
#if OCCA_MMX
  union {
    __m64 reg;
    float vec[2];
  };
#else
  float vec[2];
#endif
};

struct vfloat4 {
#if OCCA_SSE
  union {
    __m128 reg;
    float vec[4];
  };
#else
  float vec[4];
#endif
};

struct vfloat8 {
#if OCCA_AVX
  union {
    __m256 reg;
    float vec[4];
  };
#else
  float vec[4];
#endif
};

struct vdouble2 {
#if OCCA_SSE2
  union {
    __m128d reg;
    double vec[2];
  };
#else
  double vec[2];
#endif
};

struct vdouble4 {
#if OCCA_AVX
  union {
    __m256d reg;
    double vec[4];
  };
#else
  double vec[4];
#endif
};

#if OCCA_USING_CPU
#if OCCA_SSE
inline vfloat4 & operator += (vfloat4 & a, vfloat4 & b) {
#if OCCA_SSE

    a.reg = _mm_add_ps(a.reg, b.reg);
    return a;
#else
#endif
}
#endif
#endif

#if OCCA_USING_CPU
#if OCCA_SSE
inline vfloat4 operator + (vfloat4 & a, vfloat4 & b) {
#if OCCA_SSE

    vfloat4 ret;
    ret.reg = _mm_add_ps(a.reg, b.reg);
    return ret;
#else
#endif
}
#endif
#endif

