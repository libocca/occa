#ifndef OCCA_PTHREADS_DEFINES_HEADER
#define OCCA_PTHREADS_DEFINES_HEADER

#include <stdint.h>

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include "occaBase.hpp"

//---[ Defines ]----------------------------------
#define OCCA_MAX_THREADS 512

#ifndef OCCA_MEM_ALIGN
#  define OCCA_MEM_ALIGN 16
#endif

#define OCCA_IN_KERNEL      1

#define OCCA_USING_SERIAL   0
#define OCCA_USING_OPENMP   0
#define OCCA_USING_OPENCL   0
#define OCCA_USING_CUDA     0
#define OCCA_USING_PTHREADS 1
#define OCCA_USING_COI      0

#define OCCA_USING_CPU (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS || OCCA_USING_COI)
#define OCCA_USING_GPU (OCCA_USING_OPENCL || OCCA_USING_CUDA)
//================================================


//---[ Math Defines ]-----------------------------
#define OCCA_E         2.7182818284590452 // e
#define OCCA_LOG2E     1.4426950408889634 // log2(e)
#define OCCA_LOG10E    0.4342944819032518 // log10(e)
#define OCCA_LN2       0.6931471805599453 // loge(2)
#define OCCA_LN10      2.3025850929940456 // loge(10)
#define OCCA_PI        3.1415926535897932 // pi
#define OCCA_PI_2      1.5707963267948966 // pi/2
#define OCCA_PI_4      0.7853981633974483 // pi/4
#define OCCA_1_PI      0.3183098861837906 // 1/pi
#define OCCA_2_PI      0.6366197723675813 // 2/pi
#define OCCA_2_SQRTPI  1.1283791670955125 // 2/sqrt(pi)
#define OCCA_SQRT2     1.4142135623730950 // sqrt(2)
#define OCCA_SQRT1_2   0.7071067811865475 // 1/sqrt(2)
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
#define occaOuterStart2 occaKernelArgs[6]
#define occaOuterEnd2   occaKernelArgs[7]
#define occaOuterStart1 occaKernelArgs[8]
#define occaOuterEnd1   occaKernelArgs[9]
#define occaOuterStart0 occaKernelArgs[10]
#define occaOuterEnd0   occaKernelArgs[11]
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaGlobalDim2 (occaInnerDim2 * occaOuterDim2)
#define occaGlobalId2  (occaOuterId2*occaInnerDim2 + occaInnerId2)

#define occaGlobalDim1 (occaInnerDim1 * occaOuterDim1)
#define occaGlobalId1  (occaOuterId1*occaInnerDim1 + occaInnerId1)

#define occaGlobalDim0 (occaInnerDim0 * occaOuterDim0)
#define occaGlobalId0  (occaOuterId0*occaInnerDim0 + occaInnerId0)
//================================================


//---[ Loops ]------------------------------------
#define occaOuterFor2 for(int occaOuterId2 = occaOuterStart2; occaOuterId2 < occaOuterEnd2; ++occaOuterId2)
#define occaOuterFor1 for(int occaOuterId1 = occaOuterStart1; occaOuterId1 < occaOuterEnd1; ++occaOuterId1)
#define occaOuterFor0 for(int occaOuterId0 = occaOuterStart0; occaOuterId0 < occaOuterEnd0; ++occaOuterId0)

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
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaDirectLoad(X) (*(X))
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
//================================================


//---[ Atomics ]----------------------------------
// [-] Not implemented yet
template <class TM>
TM occaAtomicAdd(TM *ptr, const TM &update){
  const TM old = *ptr;

  *ptr += update;

  return old;
}

template <class TM>
TM occaAtomicSub(TM *ptr, const TM &update){
  const TM old = *ptr;

  *ptr -= update;

  return old;
}

template <class TM>
TM occaAtomicSwap(TM *ptr, const TM &update){
  const TM old = *ptr;

  *ptr = update;

  return old;
}

template <class TM>
TM occaAtomicInc(TM *ptr){
  const TM old = *ptr;

  ++(*ptr);

  return old;
}

template <class TM>
TM occaAtomicDec(TM *ptr, const TM &update){
  const TM old = *ptr;

  --(*ptr);

  return old;
}

template <class TM>
TM occaAtomicMin(TM *ptr, const TM &update){
  const TM old = *ptr;

  *ptr = ((old < update) ? old : update);

  return old;
}

template <class TM>
TM occaAtomicAnd(TM *ptr, const TM &update){
  const TM old = *ptr;

  *ptr &= update;

  return old;
}

template <class TM>
TM occaAtomicOr(TM *ptr, const TM &update){
  const TM old = *ptr;

  *ptr |= update;

  return old;
}

template <class TM>
TM occaAtomicXor(TM *ptr, const TM &update){
  const TM old = *ptr;

  *ptr ^= update;

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
#define occaParallelFor2
#define occaParallelFor1
#define occaParallelFor0
#define occaParallelFor
// - - - - - - - - - - - - - - - - - - - - - - - -
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
