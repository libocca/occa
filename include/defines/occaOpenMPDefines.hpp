#ifndef OCCA_OPENMP_DEFINES_HEADER
#define OCCA_OPENMP_DEFINES_HEADER

#include <stdint.h>

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include "occaBase.hpp"

//---[ Defines ]----------------------------------
#define OCCA_MAX_THREADS 512
#define OCCA_MEM_ALIGN   64

template <class TM>
class type2 {
public:
  TM x,y;

  inline type2(TM x_ = 0, TM y_ = 0) :
    x(x_), y(y_) {}

  inline type2<TM> operator - () const {
    return type2<TM>(-x, -y);
  }

  template <class TM2> type2<TM>& operator  = (const type2<TM2> &t);
  template <class TM2> type2<TM>& operator += (const type2<TM2> &t);
  template <class TM2> type2<TM>& operator -= (const type2<TM2> &t);
  template <class TM2> type2<TM>& operator *= (const type2<TM2> &t);
  template <class TM2> type2<TM>& operator /= (const type2<TM2> &t);

  template <class TM2> type2<TM>& operator  = (const TM2 &t);
  template <class TM2> type2<TM>& operator += (const TM2 &t);
  template <class TM2> type2<TM>& operator -= (const TM2 &t);
  template <class TM2> type2<TM>& operator *= (const TM2 &t);
  template <class TM2> type2<TM>& operator /= (const TM2 &t);
};

template <class TM>
class type3 {
public:
  TM x,y,z;

  inline type3(TM x_ = 0, TM y_ = 0, TM z_ = 0) :
    x(x_), y(y_), z(z_) {}

  template <class TM2> type3<TM>& operator  = (const type3<TM2> &t);
  template <class TM2> type3<TM>& operator += (const type3<TM2> &t);
  template <class TM2> type3<TM>& operator -= (const type3<TM2> &t);
  template <class TM2> type3<TM>& operator *= (const type3<TM2> &t);
  template <class TM2> type3<TM>& operator /= (const type3<TM2> &t);

  template <class TM2> type3<TM>& operator  = (const TM2 &t);
  template <class TM2> type3<TM>& operator += (const TM2 &t);
  template <class TM2> type3<TM>& operator -= (const TM2 &t);
  template <class TM2> type3<TM>& operator *= (const TM2 &t);
  template <class TM2> type3<TM>& operator /= (const TM2 &t);
};

template <class TM>
class type4 {
public:
  TM x,y,z,w;

  inline type4(TM x_ = 0, TM y_ = 0, TM z_ = 0, TM w_ = 0) :
    x(x_), y(y_), z(z_), w(w_) {}

  inline type4<TM> operator - () const {
    return type4<TM>(-x, -y, -z, -w);
  }

  template <class TM2> type4<TM>& operator  = (const type4<TM2> &t);
  template <class TM2> type4<TM>& operator += (const type4<TM2> &t);
  template <class TM2> type4<TM>& operator -= (const type4<TM2> &t);
  template <class TM2> type4<TM>& operator *= (const type4<TM2> &t);
  template <class TM2> type4<TM>& operator /= (const type4<TM2> &t);

  template <class TM2> type4<TM>& operator  = (const TM2 &t);
  template <class TM2> type4<TM>& operator += (const TM2 &t);
  template <class TM2> type4<TM>& operator -= (const TM2 &t);
  template <class TM2> type4<TM>& operator *= (const TM2 &t);
  template <class TM2> type4<TM>& operator /= (const TM2 &t);
};

template <class TM>
class type8 {
public:
  TM x,y,z,w,s4,s5,s6,s7;

  inline type8(TM x_  = 0, TM y_  = 0, TM z_  = 0, TM w_  = 0,
               TM s4_ = 0, TM s5_ = 0, TM s6_ = 0, TM s7_ = 0) :
    x(x_)  , y(y_)    , z(z_)  , w(w_),
    s4(s4_),  s5(s5_),  s6(s6_), s7(s7_) {}

  inline type8<TM> operator - () const {
    return type8<TM>(-x , -y , -w , -z,
                     -s4, -s5, -s6, -s7);
  }

  template <class TM2> type8<TM>& operator  = (const type8<TM2> &t);
  template <class TM2> type8<TM>& operator += (const type8<TM2> &t);
  template <class TM2> type8<TM>& operator -= (const type8<TM2> &t);
  template <class TM2> type8<TM>& operator *= (const type8<TM2> &t);
  template <class TM2> type8<TM>& operator /= (const type8<TM2> &t);

  template <class TM2> type8<TM>& operator  = (const TM2 &t);
  template <class TM2> type8<TM>& operator += (const TM2 &t);
  template <class TM2> type8<TM>& operator -= (const TM2 &t);
  template <class TM2> type8<TM>& operator *= (const TM2 &t);
  template <class TM2> type8<TM>& operator /= (const TM2 &t);
};

template <class TM>
class type16 {
public:
  TM x,y,z,w,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15;

  inline type16(TM x_   = 0, TM y_   = 0, TM z_   = 0, TM w_   = 0,
                TM s4_  = 0, TM s5_  = 0, TM s6_  = 0, TM s7_  = 0,
                TM s8_  = 0, TM s9_  = 0, TM s10_ = 0, TM s11_ = 0,
                TM s12_ = 0, TM s13_ = 0, TM s14_ = 0, TM s15_ = 0) :
    x(x_)    , y(y_)      , z(z_)    , w(w_)     ,
    s4(s4_)  ,  s5(s5_)  ,  s6(s6_)  ,  s7(s7_)  ,
    s8(s8_)  ,  s9(s9_)  ,  s10(s10_),  s11(s11_),
    s12(s12_),  s13(s13_),  s14(s14_),  s15(s15_) {}

  inline type16<TM> operator - () const {
    return type16<TM>(-x  , -y  , -w  , -z  ,
                      -s4 , -s5 , -s6 , -s7 ,
                      -s8 , -s9 , -s10, -s11,
                      -s12, -s13, -s14, -s15);
  }

  template <class TM2> type16<TM>& operator  = (const type16<TM2> &t);
  template <class TM2> type16<TM>& operator += (const type16<TM2> &t);
  template <class TM2> type16<TM>& operator -= (const type16<TM2> &t);
  template <class TM2> type16<TM>& operator *= (const type16<TM2> &t);
  template <class TM2> type16<TM>& operator /= (const type16<TM2> &t);

  template <class TM2> type16<TM>& operator  = (const TM2 &t);
  template <class TM2> type16<TM>& operator += (const TM2 &t);
  template <class TM2> type16<TM>& operator -= (const TM2 &t);
  template <class TM2> type16<TM>& operator *= (const TM2 &t);
  template <class TM2> type16<TM>& operator /= (const TM2 &t);
};

#define OCCA_TYPE_OPERATOR(O)                                           \
  template <class TM>                                                   \
  type2<TM> operator O (const type2<TM> &a, const type2<TM> &b){        \
    return type2<TM>((a.x O b.x), (a.y O b.y));                         \
  }                                                                     \
  template <class TM>                                                   \
  type2<TM> operator O (const type2<TM> &a, const TM &b){               \
    return type2<TM>((a.x O b), (a.y O b));                             \
  }                                                                     \
  template <class TM>                                                   \
  type2<TM> operator O (const TM &b, const type2<TM> &a){               \
    return type2<TM>((b O a.x), (b O a.y));                             \
  }                                                                     \
                                                                        \
  template <class TM>                                                   \
  type3<TM> operator O (const type3<TM> &a, const type3<TM> &b){        \
    return type3<TM>((a.x O b.x), (a.y O b.y), (a.z O b.z));            \
  }                                                                     \
  template <class TM>                                                   \
  type3<TM> operator O (const type3<TM> &a, const TM &b){               \
    return type3<TM>((a.x O b), (a.y O b), (a.z O b));                  \
  }                                                                     \
  template <class TM>                                                   \
  type3<TM> operator O (const TM &b, const type3<TM> &a){               \
    return type3<TM>((b O a.x), (b O a.y), (b O a.z));                  \
  }                                                                     \
                                                                        \
  template <class TM>                                                   \
  type4<TM> operator O (const type4<TM> &a, const type4<TM> &b){        \
    return type4<TM>((a.x O b.x), (a.y O b.y), (a.z O b.z), (a.w O b.w)); \
  }                                                                     \
  template <class TM>                                                   \
  type4<TM> operator O (const type4<TM> &a, const TM &b){               \
    return type4<TM>((a.x O b), (a.y O b), (a.z O b), (a.w O b));       \
  }                                                                     \
  template <class TM>                                                   \
  type4<TM> operator O (const TM &b, const type4<TM> &a){               \
    return type4<TM>((b O a.x), (b O a.y), (b O a.z), (b O a.w));       \
  }                                                                     \
                                                                        \
  template <class TM>                                                   \
  type8<TM> operator O (const type8<TM> &a, const type8<TM> &b){        \
    return type8<TM>((a.x  O b.x) , (a.y  O b.y) , (a.z  O b.z) , (a.w  O b.w) , \
                     (a.s4 O b.s4), (a.s5 O b.s5), (a.s6 O b.s6), (a.s7 O b.s7)); \
  }                                                                     \
  template <class TM>                                                   \
  type8<TM> operator O (const type8<TM> &a, const TM &b){               \
    return type8<TM>((a.x  O b), (a.y  O b), (a.z  O b), (a.w  O b) ,   \
                     (a.s4 O b), (a.s5 O b), (a.s6 O b), (a.s7 O b));   \
  }                                                                     \
  template <class TM>                                                   \
  type8<TM> operator O (const TM &b, const type8<TM> &a){               \
    return type8<TM>((b O a.x ), (b O a.y ), (b O a.z ), (b O a.w ) ,   \
                     (b O a.s4), (b O a.s5), (b O a.s6), (b O a.s7));   \
  }                                                                     \
                                                                        \
  template <class TM>                                                   \
  type16<TM> operator O (const type16<TM> &a, const type16<TM> &b){     \
    return type16<TM>((a.x   O b.x)  , (a.y   O b.y)  , (a.z   O b.z)  , (a.w   O b.w)  , \
                      (a.s4  O b.s4) , (a.s5  O b.s5) , (a.s6  O b.s6) , (a.s7  O b.s7) , \
                      (a.s8  O b.s8) , (a.s9  O b.s9) , (a.s10 O b.s10), (a.s11 O b.s11), \
                      (a.s12 O b.s12), (a.s13 O b.s13), (a.s14 O b.s14), (a.s15 O b.s15)); \
  }                                                                     \
  template <class TM>                                                   \
  type16<TM> operator O (const type16<TM> &a, const TM &b){             \
    return type16<TM>((a.x   O b), (a.y   O b), (a.z   O b), (a.w   O b), \
                      (a.s4  O b), (a.s5  O b), (a.s6  O b), (a.s7  O b), \
                      (a.s8  O b), (a.s9  O b), (a.s10 O b), (a.s11 O b), \
                      (a.s12 O b), (a.s13 O b), (a.s14 O b), (a.s15 O b)); \
  }                                                                     \
  template <class TM>                                                   \
  type16<TM> operator O (const TM &b, const type16<TM> &a){             \
    return type16<TM>((b O a.x  ), (b O a.y  ), (b O a.z  ), (b O a.w  ), \
                      (b O a.s4 ), (b O a.s5 ), (b O a.s6 ), (b O a.s7 ), \
                      (b O a.s8 ), (b O a.s9 ), (b O a.s10), (b O a.s11), \
                      (b O a.s12), (b O a.s13), (b O a.s14), (b O a.s15)); \
  }

#define OCCA_TYPE_EQUAL_OPERATOR(O)                           \
  template <class TM>                                         \
  template <class TM2>                                        \
  type2<TM>& type2<TM>::operator O (const type2<TM2> &t){     \
    x O t.x; y O t.y;                                         \
    return *this;                                             \
  }                                                           \
  template <class TM>                                         \
  template <class TM2>                                        \
  type2<TM>& type2<TM>::operator O (const TM2 &t){            \
    x O t; y O t;                                             \
    return *this;                                             \
  }                                                           \
                                                              \
  template <class TM>                                         \
  template <class TM2>                                        \
  type3<TM>& type3<TM>::operator O (const type3<TM2> &t){     \
    x O t.x; y O t.y; z O t.z;                                \
    return *this;                                             \
  }                                                           \
  template <class TM>                                         \
  template <class TM2>                                        \
  type3<TM>& type3<TM>::operator O (const TM2 &t){            \
    x O t; y O t; z O t;                                      \
    return *this;                                             \
  }                                                           \
                                                              \
  template <class TM>                                         \
  template <class TM2>                                        \
  type4<TM>& type4<TM>::operator O (const type4<TM2> &t){     \
    x O t.x; y O t.y; z O t.z; w O t.w;                       \
    return *this;                                             \
  }                                                           \
  template <class TM>                                         \
  template <class TM2>                                        \
  type4<TM>& type4<TM>::operator O (const TM2 &t){             \
    x O t; y O t; z O t; w O t;                               \
    return *this;                                             \
  }                                                           \
                                                              \
  template <class TM>                                         \
  template <class TM2>                                        \
  type8<TM>& type8<TM>::operator O (const type8<TM2> &t){     \
    x  O t.x;  y  O t.y;  z  O t.z;  w  O t.w;                \
    s4 O t.s4; s5 O t.s5; s6 O t.s6; s7 O t.s7;               \
    return *this;                                             \
  }                                                           \
  template <class TM>                                         \
  template <class TM2>                                        \
  type8<TM>& type8<TM>::operator O (const TM2 &t){            \
    x  O t; y  O t; z  O t; w  O t;                           \
    s4 O t; s5 O t; s6 O t; s7 O t;                           \
    return *this;                                             \
  }                                                           \
                                                              \
  template <class TM>                                         \
  template <class TM2>                                        \
  type16<TM>& type16<TM>::operator O (const type16<TM2> &t){  \
    x   O t.x;   y  O t.y;    z   O t.z;   w   O t.w;         \
    s4  O t.s4;  s5 O t.s5;   s6  O t.s6;  s7  O t.s7;        \
    s8  O t.s8;  s9 O t.s9;   s10 O t.s10; s11 O t.s11;       \
    s12 O t.s12; s13 O t.s13; s14 O t.s14; s15 O t.s15;       \
    return *this;                                             \
  }                                                           \
  template <class TM>                                         \
  template <class TM2>                                        \
  type16<TM>& type16<TM>::operator O (const TM2 &t){          \
    x   O t; y  O t;  z   O t; w   O t;                       \
    s4  O t; s5 O t;  s6  O t; s7  O t;                       \
    s8  O t; s9 O t;  s10 O t; s11 O t;                       \
    s12 O t; s13 O t; s14 O t; s15 O t;                       \
    return *this;                                             \
  }

OCCA_TYPE_OPERATOR(+);
OCCA_TYPE_OPERATOR(-);
OCCA_TYPE_OPERATOR(*);
OCCA_TYPE_OPERATOR(/);

OCCA_TYPE_EQUAL_OPERATOR(=);
OCCA_TYPE_EQUAL_OPERATOR(+=);
OCCA_TYPE_EQUAL_OPERATOR(-=);
OCCA_TYPE_EQUAL_OPERATOR(*=);
OCCA_TYPE_EQUAL_OPERATOR(/=);

typedef type2<char>  char2;
typedef type3<char>  char3;
typedef type4<char>  char4;
typedef type8<char>  char8;
typedef type16<char> char16;

typedef type2<short>  short2;
typedef type3<short>  short3;
typedef type4<short>  short4;
typedef type8<short>  short8;
typedef type16<short> short16;

typedef type2<int>  int2;
typedef type3<int>  int3;
typedef type4<int>  int4;
typedef type8<int>  int8;
typedef type16<int> int16;

typedef type2<float>  float2;
typedef type3<float>  float3;
typedef type4<float>  float4;
typedef type8<float>  float8;
typedef type16<float> float16;

typedef type2<double>  double2;
typedef type3<double>  double3;
typedef type4<double>  double4;
typedef type8<double>  double8;
typedef type16<double> double16;
//================================================


//---[ Loop Info ]--------------------------------
#define occaOuterDim2 occaKernelArgs[0]
#define occaOuterDim1 occaKernelArgs[1]
#define occaOuterDim0 occaKernelArgs[2]
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaInnerDim2 occaKernelArgs[3]
#define occaInnerDim1 occaKernelArgs[4]
#define occaInnerDim0 occaKernelArgs[5]
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaGlobalDim2 (occaInnerDim2 * occaOuterDim2)
#define occaGlobalId2  (occaOuterId2*occaInnerDim2 + occaInnerId2)

#define occaGlobalDim1 (occaInnerDim1 * occaOuterDim1)
#define occaGlobalId1  (occaOuterId1*occaInnerDim1 + occaInnerId1)

#define occaGlobalDim0 (occaInnerDim0 * occaOuterDim0)
#define occaGlobalId0  (occaOuterId0*occaInnerDim0 + occaInnerId0)
//================================================


//---[ Loops ]------------------------------------
#define occaOuterFor2 for(int occaOuterId2 = 0; occaOuterId2 < occaOuterDim2; ++occaOuterId2)
#define occaOuterFor1 for(int occaOuterId1 = 0; occaOuterId1 < occaOuterDim1; ++occaOuterId1)
#define occaOuterFor0 for(int occaOuterId0 = 0; occaOuterId0 < occaOuterDim0; ++occaOuterId0)

#define occaOuterFor occaOuterFor2 occaOuterFor1 occaOuterFor0
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaInnerFor2 for(occaInnerId2 = 0; occaInnerId2 < occaInnerDim2; ++occaInnerId2)
#define occaInnerFor1 for(occaInnerId1 = 0; occaInnerId1 < occaInnerDim1; ++occaInnerId1)
#define occaInnerFor0 for(occaInnerId0 = 0; occaInnerId0 < occaInnerDim0; ++occaInnerId0)
#define occaInnerFor occaInnerFor2 occaInnerFor1 occaInnerFor0
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaGlobalFor0 occaOuterFor0 occaInnerFor0
//================================================


//---[ Standard Functions ]-----------------------
#define occaLocalMemFence
#define occaGlobalMemFence

#define occaBarrier(FENCE)
#define occaInnerBarrier(FENCE) continue
#define occaOuterBarrier(FENCE)
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaContinue continue
//================================================


//---[ Attributes ]-------------------------------
#define occaShared
#define occaPointer
#define occaVariable &

#ifndef MC_CL_EXE
#  define occaRestrict __restrict__
#  define occaVolatile volatile
#  define occaAligned  __attribute__ ((aligned (OCCA_MEM_ALIGN)))
#else
// branch for Microsoft cl.exe - compiler: __restrict__ and __attribute__ ((aligned(...))) are not available there.
#  define occaRestrict
// [dsm5] Volatile doesn't work on WIN, it's not that important anyway (for now)
#  define occaVolatile
#  define occaAligned
#endif

#define occaFunctionShared
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaConst    const
#define occaConstant const
//================================================


//---[ Kernel Info ]------------------------------
#define occaKernelInfoArg   const int *occaKernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2
#define occaFunctionInfoArg const int *occaKernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2
#define occaFunctionInfo               occaKernelArgs,     occaInnerId0,     occaInnerId1,     occaInnerId2
// - - - - - - - - - - - - - - - - - - - - - - - -
#ifndef MC_CL_EXE
#  define occaKernel extern "C"
#else
// branch for Microsoft cl.exe - compiler: each symbol that a dll (shared object) should export must be decorated with __declspec(dllexport)
#  define occaKernel extern "C" __declspec(dllexport)
#endif

#define occaFunction
#define occaDeviceFunction

#ifndef MC_CL_EXE
#  define OCCA_PRAGMA(STR) _Pragma(STR)
#else
#  define OCCA_PRAGMA(STR) __pragma(STR)
#endif

#if OCCA_OPENMP_ENABLED
#  define OCCA_OMP_PRAGMA(STR) OCCA_PRAGMA(STR)
#else
#  define OCCA_OMP_PRAGMA(STR)
#endif
//================================================


//---[ Atomics ]----------------------------------
template <class TM>
TM occaAtomicAdd(TM *ptr, const TM &update){
#if !OCCA_OPENMP_ENABLED
  const TM old = *ptr;
  *ptr += update;
#else
  TM old;

#  ifdef OPENMP_3_1
#    pragma omp atomic capture
#  else
#    pragma omp critical
#  endif
  {
    old   = *ptr;
    *ptr += update;
  }
#endif

  return old;
}

template <class TM>
TM occaAtomicSub(TM *ptr, const TM &update){
#if !OCCA_OPENMP_ENABLED
  const TM old = *ptr;
  *ptr -= update;
#else
  TM old;

#  ifdef OPENMP_3_1
#    pragma omp atomic capture
#  else
#    pragma omp critical
#  endif
  {
    old   = *ptr;
    *ptr -= update;
  }
#endif

  return old;
}

template <class TM>
TM occaAtomicSwap(TM *ptr, const TM &update){
#if !OCCA_OPENMP_ENABLED
  const TM old = *ptr;
  *ptr = update;
#else
  TM old;

#  ifdef OPENMP_3_1
#    pragma omp atomic capture
#  else
#    pragma omp critical
#  endif
  {
    old  = *ptr;
    *ptr = update;
  }
#endif

  return old;
}

template <class TM>
TM occaAtomicInc(TM *ptr){
#if !OCCA_OPENMP_ENABLED
  const TM old = *ptr;
  ++(*ptr);
#else
  TM old;

#  ifdef OPENMP_3_1
#    pragma omp atomic capture
#  else
#    pragma omp critical
#  endif
  {
    old = *ptr;
  ++(*ptr);
  }
#endif

  return old;
}

template <class TM>
TM occaAtomicDec(TM *ptr, const TM &update){
#if !OCCA_OPENMP_ENABLED
  const TM old = *ptr;
  --(*ptr);
#else
  TM old;

#  ifdef OPENMP_3_1
#    pragma omp atomic capture
#  else
#    pragma omp critical
#  endif
  {
    old = *ptr;
    --(*ptr);
  }
#endif

  return old;
}

template <class TM>
TM occaAtomicMin(TM *ptr, const TM &update){
#if !OCCA_OPENMP_ENABLED
  const TM old = *ptr;
  *ptr = ((old < update) ? old : update);
#else
  TM old;

#  pragma omp critical
  {
    old  = *ptr;
    *ptr = ((old < update) ? old : update);
  }
#endif

  return old;
}

template <class TM>
TM occaAtomicMax(TM *ptr, const TM &update){
#if !OCCA_OPENMP_ENABLED
  const TM old = *ptr;
  *ptr = ((old < update) ? update : old);
#else
  TM old;

#  pragma omp critical
  {
    old  = *ptr;
    *ptr = ((old < update) ? update : old);
  }
#endif

  return old;
}

template <class TM>
TM occaAtomicAnd(TM *ptr, const TM &update){
#if !OCCA_OPENMP_ENABLED
  const TM old = *ptr;
  *ptr &= update;
#else
  TM old;

#  ifdef OPENMP_3_1
#    pragma omp atomic capture
#  else
#    pragma omp critical
#  endif
  {
    old   = *ptr;
    *ptr &= update;
  }
#endif

  return old;
}

template <class TM>
TM occaAtomicOr(TM *ptr, const TM &update){
#if !OCCA_OPENMP_ENABLED
  const TM old = *ptr;
  *ptr |= update;
#else
  TM old;

#  ifdef OPENMP_3_1
#    pragma omp atomic capture
#  else
#    pragma omp critical
#  endif
  {
    old   = *ptr;
    *ptr |= update;
  }
#endif

  return old;
}

template <class TM>
TM occaAtomicXor(TM *ptr, const TM &update){
#if !OCCA_OPENMP_ENABLED
  const TM old = *ptr;
  *ptr ^= update;
#else
  TM old;

#  ifdef OPENMP_3_1
#    pragma omp atomic capture
#  else
#    pragma omp critical
#  endif
  {
    old   = *ptr;
    *ptr ^= update;
  }
#endif

  return old;
}

#define occaAtomicAdd64  occaAtomicAdd
#define occaAtomicSub64  occaAtomicSub
#define occaAtomicSwap64 occaAtomicSwap
#define occaAtomicInc64  occaAtomicInc
#define occaAtomicDec64  occaAtomicDec
//================================================


//---[ Math ]-------------------------------------
#define occaFabs       fabs
#define occaFastFabs   fabs
#define occaNativeFabs fabs

#define occaSqrt       sqrt
#define occaFastSqrt   sqrt
#define occaNativeSqrt sqrt

#define occaCbrt       cbrt
#define occaFastCbrt   cbrt
#define occaNativeCbrt cbrt

#define occaSin       sin
#define occaFastSin   sin
#define occaNativeSin sin

#define occaAsin       asin
#define occaFastAsin   asin
#define occaNativeAsin asin

#define occaSinh       sinh
#define occaFastSinh   sinh
#define occaNativeSinh sinh

#define occaAsinh       asinh
#define occaFastAsinh   asinh
#define occaNativeAsinh asinh

#define occaCos       cos
#define occaFastCos   cos
#define occaNativeCos cos

#define occaAcos       acos
#define occaFastAcos   acos
#define occaNativeAcos acos

#define occaCosh       cosh
#define occaFastCosh   cosh
#define occaNativeCosh cosh

#define occaAcosh       acosh
#define occaFastAcosh   acosh
#define occaNativeAcosh acosh

#define occaTan       tan
#define occaFastTan   tan
#define occaNativeTan tan

#define occaAtan       atan
#define occaFastAtan   atan
#define occaNativeAtan atan

#define occaTanh       tanh
#define occaFastTanh   tanh
#define occaNativeTanh tanh

#define occaAtanh       atanh
#define occaFastAtanh   atanh
#define occaNativeAtanh atanh

#define occaExp       exp
#define occaFastExp   exp
#define occaNativeExp exp

#define occaExpm1       expm1
#define occaFastExpm1   expm1
#define occaNativeExpm1 expm1

#define occaPow       pow
#define occaFastPow   pow
#define occaNativePow pow

#define occaLog2       log2
#define occaFastLog2   log2
#define occaNativeLog2 log2

#define occaLog10       log10
#define occaFastLog10   log10
#define occaNativeLog10 log10
//================================================


//---[ Misc ]-------------------------------------
#define occaParallelFor2 OCCA_OMP_PRAGMA("omp parallel for collapse(3) firstprivate(occaInnerId0,occaInnerId1,occaInnerId2)")
#define occaParallelFor1 OCCA_OMP_PRAGMA("omp parallel for collapse(2) firstprivate(occaInnerId0,occaInnerId1,occaInnerId2)")
#define occaParallelFor0 OCCA_OMP_PRAGMA("omp parallel for             firstprivate(occaInnerId0,occaInnerId1,occaInnerId2)")
#define occaParallelFor  OCCA_OMP_PRAGMA("omp parallel for             firstprivate(occaInnerId0,occaInnerId1,occaInnerId2)")
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaUnroll3(N) OCCA_PRAGMA(#N)
#define occaUnroll2(N) occaUnroll3(N)
#define occaUnroll(N)  occaUnroll2(unroll N)
//================================================


//---[ Private ]---------------------------------
template <class TM, const int SIZE>
class occaPrivate_t {
public:
  const int dim0, dim1, dim2;
  const int &id0, &id1, &id2;

  TM data[OCCA_MAX_THREADS][SIZE] occaAligned;

  occaPrivate_t(int dim0_, int dim1_, int dim2_,
                int &id0_, int &id1_, int &id2_) :
    dim0(dim0_),
    dim1(dim1_),
    dim2(dim2_),
    id0(id0_),
    id1(id1_),
    id2(id2_) {}

  ~occaPrivate_t(){}

  inline int index() const {
    return ((id2*dim1 + id1)*dim0 + id0);
  }

  inline TM& operator [] (const int n){
    return data[index()][n];
  }

  inline TM operator [] (const int n) const {
    return data[index()][n];
  }

  inline operator TM(){
    return data[index()][0];
  }

  inline operator TM*(){
    return data[index()];
  }

  inline TM& operator = (const occaPrivate_t &r) {
    data[index()][0] = r.data[index()][0];
    return data[index()][0];
  }

  inline TM& operator = (const TM &t){
    data[index()][0] = t;
    return data[index()][0];
  }

  inline TM& operator += (const TM &t){
    data[index()][0] += t;
    return data[index()][0];
  }

  inline TM& operator -= (const TM &t){
    data[index()][0] -= t;
    return data[index()][0];
  }

  inline TM& operator /= (const TM &t){
    data[index()][0] /= t;
    return data[index()][0];
  }

  inline TM& operator *= (const TM &t){
    data[index()][0] *= t;
    return data[index()][0];
  }

  friend inline TM operator + (const TM &a, const occaPrivate_t &b){
    return (a + b.data[b.index()][0]);
  }

  friend inline TM operator + (const occaPrivate_t &a, const TM &b){
    return (a.data[a.index()][0] + b);
  }

  friend inline TM operator - (const TM &a, const occaPrivate_t &b){
    return (a - b.data[b.index()][0]);
  }

  friend inline TM operator - (const occaPrivate_t &a, const TM &b){
    return (a.data[a.index()][0] - b);
  }

  friend inline TM operator * (const TM &a, const occaPrivate_t &b){
    return (a * b.data[b.index()][0]);
  }

  friend inline TM operator * (const occaPrivate_t &a, const TM &b){
    return (a.data[a.index()][0] * b);
  }

  friend inline TM operator / (const TM &a, const occaPrivate_t &b){
    return (a / b.data[b.index()][0]);
  }

  friend inline TM operator / (const occaPrivate_t &a, const TM &b){
    return (a.data[a.index()][0] / b);
  }

  inline TM& operator ++ (){
    return (++data[index()][0]);
  }

  inline TM& operator ++ (int){
    return (data[index()][0]++);
  }

  inline TM& operator -- (){
    return (--data[index()][0]);
  }

  inline TM& operator -- (int){
    return (data[index()][0]--);
  }
};

#define occaPrivateArray( TYPE , NAME , SIZE )                          \
  occaPrivate_t<TYPE,SIZE> NAME(occaInnerDim0, occaInnerDim1, occaInnerDim2, \
                                occaInnerId0, occaInnerId1, occaInnerId2);

#define occaPrivate( TYPE , NAME )                                      \
  occaPrivate_t<TYPE,1> NAME(occaInnerDim0, occaInnerDim1, occaInnerDim2, \
                             occaInnerId0, occaInnerId1, occaInnerId2);
//================================================


//---[ Texture ]----------------------------------
struct occaTexture {
  void *data;
  int dim;

  uintptr_t w, h, d;
};

#define occaReadOnly  const
#define occaWriteOnly

#define occaTexture1D(TEX) occaTexture &TEX
#define occaTexture2D(TEX) occaTexture &TEX

#define occaTexGet1D(TEX, TYPE, VALUE, X)    VALUE = ((TYPE*) TEX.data)[X]
#define occaTexGet2D(TEX, TYPE, VALUE, X, Y) VALUE = ((TYPE*) TEX.data)[(Y * TEX.w) + X]

#define occaTexSet1D(TEX, TYPE, VALUE, X)    ((TYPE*) TEX.data)[X]               = VALUE
#define occaTexSet2D(TEX, TYPE, VALUE, X, Y) ((TYPE*) TEX.data)[(Y * TEX.w) + X] = VALUE
//================================================

#endif
