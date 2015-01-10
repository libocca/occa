#ifndef OCCA_COI_DEFINES_HEADER
#define OCCA_COI_DEFINES_HEADER

#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <cmath>

#include <omp.h>

#include <intel-coi/sink/COIPipeline_sink.h>
#include <intel-coi/sink/COIProcess_sink.h>
#include <intel-coi/sink/COIBuffer_sink.h>
#include <intel-coi/common/COIMacros_common.h>


//---[ Defines ]----------------------------------
#define OCCA_MAX_THREADS 512
#define OCCA_MEM_ALIGN   64

typedef struct char2_t  { char x,y;                                               } char2;
typedef struct char3_t  { char x,y,z;                                             } char3;
typedef struct char4_t  { char x,y,z,w;                                           } char4;
typedef struct char8_t  { char x,y,z,w,s4,s5,s6,s7;                               } char8;
typedef struct char16_t { char x,y,z,w,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15; } char16;

typedef struct short2_t  { short x,y;                                               } short2;
typedef struct short3_t  { short x,y,z;                                             } short3;
typedef struct short4_t  { short x,y,z,w;                                           } short4;
typedef struct short8_t  { short x,y,z,w,s4,s5,s6,s7;                               } short8;
typedef struct short16_t { short x,y,z,w,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15; } short16;

typedef struct int2_t  { int x,y;                                               } int2;
typedef struct int3_t  { int x,y,z;                                             } int3;
typedef struct int4_t  { int x,y,z,w;                                           } int4;
typedef struct int8_t  { int x,y,z,w,s4,s5,s6,s7;                               } int8;
typedef struct int16_t { int x,y,z,w,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15; } int16;

typedef struct float2_t  { float x,y;                                               } float2;
typedef struct float3_t  { float x,y,z;                                             } float3;
typedef struct float4_t  { float x,y,z,w;                                           } float4;
typedef struct float8_t  { float x,y,z,w,s4,s5,s6,s7;                               } float8;
typedef struct float16_t { float x,y,z,w,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15; } float16;

typedef struct double2_t  { double x,y;                                               } double2;
typedef struct double3_t  { double x,y,z;                                             } double3;
typedef struct double4_t  { double x,y,z,w;                                           } double4;
typedef struct double8_t  { double x,y,z,w,s4,s5,s6,s7;                               } double8;
typedef struct double16_t { double x,y,z,w,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15; } double16;
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
#define occaKernel         COINATIVELIBEXPORT
#define occaFunction
#define occaDeviceFunction
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
#define occaParallelFor2 _Pragma("omp parallel for collapse(3) schedule(static)")
#define occaParallelFor1 _Pragma("omp parallel for collapse(2) schedule(static)")
#define occaParallelFor0 _Pragma("omp parallel for             schedule(static)")
#define occaParallelFor  _Pragma("omp parallel for             schedule(static)")
// - - - - - - - - - - 1 - - - - - - - - - - - - -
#define occaUnroll3(N0 _Pragma(#N)
#define occaUnroll3(N) _Pragma(#N)
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
