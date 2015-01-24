#ifndef OCCA_CUDA_DEFINES_HEADER
#define OCCA_CUDA_DEFINES_HEADER

//---[ Defines ]----------------------------------
//================================================


//---[ Type-N ]-----------------------------------
#define OCCA_TYPE_OPERATOR(O)                   \
  OCCA_TYPE_OPERATORS(O, char)                  \
  OCCA_TYPE_OPERATORS(O, short)                 \
  OCCA_TYPE_OPERATORS(O, int)                   \
  OCCA_TYPE_OPERATORS(O, float)                 \
  OCCA_TYPE_OPERATORS(O, double)

#define OCCA_TYPE_OPERATOR(O, TYPE)                                     \
  OCCA_TYPE_OPERATOR2(O, TYPE##2, TYPE##3, TYPE##4, TYPE##8, TYPE##16)

#define OCCA_TYPE_OPERATOR2(O, TYPE2, TYPE3, TYPE4, TYPE8, TYPE16)      \
  inline TYPE2 operator O (const TYPE2 &a, const TYPE2 &b){             \
    return TYPE2((a.x O b.x), (a.y O b.y));                             \
  }                                                                     \
  template <class TM>                                                   \
  inline TYPE2 operator O (const TYPE2 &a, const TM &b){                \
    return TYPE2((a.x O b), (a.y O b));                                 \
  }                                                                     \
  template <class TM>                                                   \
  inline TYPE2 operator O (const TM &b, const TYPE2 &a){                \
    return TYPE2((b O a.x), (b O a.y));                                 \
  }                                                                     \
                                                                        \
  inline TYPE3 operator O (const TYPE3 &a, const TYPE3 &b){             \
    return TYPE3((a.x O b.x), (a.y O b.y), (a.z O b.z));                \
  }                                                                     \
  template <class TM>                                                   \
  inline TYPE3 operator O (const TYPE3 &a, const TM &b){                \
    return TYPE3((a.x O b), (a.y O b), (a.z O b));                      \
  }                                                                     \
  template <class TM>                                                   \
  inline TYPE3 operator O (const TM &b, const TYPE3 &a){                \
    return TYPE3((b O a.x), (b O a.y), (b O a.z));                      \
  }                                                                     \
                                                                        \
  inline TYPE4 operator O (const TYPE4 &a, const TYPE4 &b){             \
    return TYPE4((a.x O b.x), (a.y O b.y), (a.z O b.z), (a.w O b.w));   \
  }                                                                     \
  template <class TM>                                                   \
  inline TYPE4 operator O (const TYPE4 &a, const TM &b){                \
    return TYPE4((a.x O b), (a.y O b), (a.z O b), (a.w O b));           \
  }                                                                     \
  template <class TM>                                                   \
  inline TYPE4 operator O (const TM &b, const TYPE4 &a){                \
    return TYPE4((b O a.x), (b O a.y), (b O a.z), (b O a.w));           \
  }                                                                     \
                                                                        \
  inline TYPE8 operator O (const TYPE8 &a, const TYPE8 &b){             \
    return TYPE8((a.x  O b.x) , (a.y  O b.y) , (a.z  O b.z) , (a.w  O b.w) , \
                 (a.s4 O b.s4), (a.s5 O b.s5), (a.s6 O b.s6), (a.s7 O b.s7)); \
  }                                                                     \
  template <class TM>                                                   \
  inline TYPE8 operator O (const TYPE8 &a, const TM &b){                \
    return TYPE8((a.x  O b), (a.y  O b), (a.z  O b), (a.w  O b) ,       \
                 (a.s4 O b), (a.s5 O b), (a.s6 O b), (a.s7 O b));       \
  }                                                                     \
  template <class TM>                                                   \
  inline TYPE8 operator O (const TM &b, const TYPE8 &a){                \
    return TYPE8((b O a.x ), (b O a.y ), (b O a.z ), (b O a.w ) ,       \
                 (b O a.s4), (b O a.s5), (b O a.s6), (b O a.s7));       \
  }                                                                     \
                                                                        \
  inline TYPE16 operator O (const TYPE16 &a, const TYPE16 &b){          \
    return TYPE16((a.x   O b.x)  , (a.y   O b.y)  , (a.z   O b.z)  , (a.w   O b.w)  , \
                  (a.s4  O b.s4) , (a.s5  O b.s5) , (a.s6  O b.s6) , (a.s7  O b.s7) , \
                  (a.s8  O b.s8) , (a.s9  O b.s9) , (a.s10 O b.s10), (a.s11 O b.s11), \
                  (a.s12 O b.s12), (a.s13 O b.s13), (a.s14 O b.s14), (a.s15 O b.s15)); \
  }                                                                     \
  template <class TM>                                                   \
  inline TYPE16 operator O (const TYPE16 &a, const TM &b){              \
    return TYPE16((a.x   O b), (a.y   O b), (a.z   O b), (a.w   O b),   \
                  (a.s4  O b), (a.s5  O b), (a.s6  O b), (a.s7  O b),   \
                  (a.s8  O b), (a.s9  O b), (a.s10 O b), (a.s11 O b),   \
                  (a.s12 O b), (a.s13 O b), (a.s14 O b), (a.s15 O b));  \
  }                                                                     \
  template <class TM>                                                   \
  inline TYPE16 operator O (const TM &b, const TYPE16 &a){              \
    return TYPE16((b O a.x  ), (b O a.y  ), (b O a.z  ), (b O a.w  ),   \
                  (b O a.s4 ), (b O a.s5 ), (b O a.s6 ), (b O a.s7 ),   \
                  (b O a.s8 ), (b O a.s9 ), (b O a.s10), (b O a.s11),   \
                  (b O a.s12), (b O a.s13), (b O a.s14), (b O a.s15));  \
  }

#define OCCA_TYPE_OPERATORS(O)                  \
  OCCA_TYPE_EQUAL_OPERATOR(O, char)             \
  OCCA_TYPE_EQUAL_OPERATOR(O, short)            \
  OCCA_TYPE_EQUAL_OPERATOR(O, int)              \
  OCCA_TYPE_EQUAL_OPERATOR(O, float)            \
  OCCA_TYPE_EQUAL_OPERATOR(O, double)

#define OCCA_TYPE_EQUAL_OPERATOR(O, TYPE)                              \
  OCCA_TYPE_EQUAL_OPERATOR2(O, TYPE##2, TYPE##3, TYPE##4, TYPE##8, TYPE##16)

#define OCCA_TYPE_EQUAL_OPERATOR2(O, TYPE2, TYPE3, TYPE4, TYPE8, TYPE16) \
  template <class TM>                                           \
  inline TYPE2& operator O (const TYPE2 &a, const TYPE2 &b){    \
    a.x O b.x; a.y O b.y;                                       \
    return a;                                                   \
  }                                                             \
  template <class TM>                                           \
  inline TYPE2& operator O (const TYPE2 &a, const TM2 &b){      \
    a.x O b; a.y O b;                                           \
    return a;                                                   \
  }                                                             \
                                                                \
  template <class TM>                                           \
  inline TYPE3& operator O (const TYPE3 &a, const TYPE3 &b){    \
    a.x O b.x; a.y O b.y; a.a.z O b.z;                          \
    return a;                                                   \
  }                                                             \
  template <class TM>                                           \
  inline TYPE3& operator O (const TYPE3 &a, const TM2 &b){      \
    a.x O b; a.y O b; a.z O b;                                  \
    return a;                                                   \
  }                                                             \
                                                                \
  template <class TM>                                           \
  inline TYPE4& operator O (const TYPE4 &a, const TYPE4 &b){    \
    a.x O b.x; a.y O b.y; a.z O b.z; a.a.w O b.w;               \
    return a;                                                   \
  }                                                             \
  template <class TM>                                           \
  inline TYPE4& operator O (const TYPE4 &a, const TM2 &b){      \
    a.x O b; a.y O b; a.z O b; a.w O b;                         \
    return a;                                                   \
  }                                                             \
                                                                \
  template <class TM>                                           \
  inline TYPE8& operator O (const TYPE8 &a, const TYPE8 &b){    \
    a.x  O b.x;  a.y  O b.y;  a.z  O b.z;  a.w  O b.w;          \
    a.s4 O b.s4; a.s5 O b.s5; a.s6 O b.s6; a.s7 O b.s7;         \
    return a;                                                   \
  }                                                             \
  template <class TM>                                           \
  inline TYPE8& operator O (const TYPE8 &a, const TM2 &b){      \
    a.x  O b; a.y  O b; a.z  O b; a.w  O b;                     \
    a.s4 O b; a.s5 O b; a.s6 O b; a.s7 O b;                     \
    return a;                                                   \
  }                                                             \
                                                                \
  template <class TM>                                           \
  inline TYPE16& operator O (const TYPE16 &a, const TYPE16 &b){ \
    a.x   O b.x;  a. y  O b.y;    a.z   O b.z;   a.w   O b.w;   \
    a.s4  O b.s4;  a.s5 O b.s5;   a.s6  O b.s6;  a.s7  O b.s7;  \
    a.s8  O b.s8;  a.s9 O b.s9;   a.s10 O b.s10; a.s11 O b.s11; \
    a.s12 O b.s12; a.s13 O b.s13; a.s14 O b.s14; a.s15 O b.s15; \
    return a;                                                   \
  }                                                             \
  template <class TM>                                           \
  inline TYPE16& operator O (const TYPE16 &a, const TM2 &b){    \
    a.x   O b; a.y  O b;  a.z   O b; a.w   O b;                 \
    a.s4  O b; a.s5 O b;  a.s6  O b; a.s7  O b;                 \
    a.s8  O b; a.s9 O b;  a.s10 O b; a.s11 O b;                 \
    a.s12 O b; a.s13 O b; a.s14 O b; a.s15 O b;                 \
    return a;                                                   \
  }

OCCA_TYPE_OPERATORS(+);
OCCA_TYPE_OPERATORS(-);
OCCA_TYPE_OPERATORS(*);
OCCA_TYPE_OPERATORS(/);

OCCA_TYPE_EQUAL_OPERATORS(+=);
OCCA_TYPE_EQUAL_OPERATORS(-=);
OCCA_TYPE_EQUAL_OPERATORS(*=);
OCCA_TYPE_EQUAL_OPERATORS(/=);
//================================================


//---[ Loop Info ]--------------------------------
#define occaOuterDim2 gridDim.z
#define occaOuterId2  blockIdx.z

#define occaOuterDim1 gridDim.y
#define occaOuterId1  blockIdx.y

#define occaOuterDim0 gridDim.x
#define occaOuterId0  blockIdx.x
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaInnerDim2 blockDim.z
#define occaInnerId2  threadIdx.z

#define occaInnerDim1 blockDim.y
#define occaInnerId1  threadIdx.y

#define occaInnerDim0 blockDim.x
#define occaInnerId0  threadIdx.x
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaGlobalDim2 (occaInnerDim2 * occaOuterDim2)
#define occaGlobalId2 (occaOuterId2*occaInnerDim2 + occaInnerId2)

#define occaGlobalDim1 (occaInnerDim1 * occaOuterDim1)
#define occaGlobalId1 (occaOuterId1*occaInnerDim1 + occaInnerId1)

#define occaGlobalDim0 (occaInnerDim0 * occaOuterDim0)
#define occaGlobalId0 (occaOuterId0*occaInnerDim0 + occaInnerId0)
//================================================


//---[ Loops ]------------------------------------
#define occaOuterFor2
#define occaOuterFor1
#define occaOuterFor0
#define occaOuterFor
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaInnerFor2
#define occaInnerFor1
#define occaInnerFor0
#define occaInnerFor
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaGlobalFor0
//================================================


//---[ Standard Functions ]-----------------------
#define occaLocalMemFence
#define occaGlobalMemFence

#define occaBarrier(FENCE)      __syncthreads()
#define occaInnerBarrier(FENCE) __syncthreads()
#define occaOuterBarrier(FENCE)
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaContinue return
//================================================


//---[ Attributes ]-------------------------------
#define occaShared   __shared__
#define occaPointer
#define occaVariable
#define occaRestrict __restrict__
#define occaVolatile volatile
#define occaAligned
#define occaFunctionShared
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaConst    const
#define occaConstant __constant__
//================================================


//---[ Kernel Info ]------------------------------
#define occaKernelInfoArg   int occaKernelInfoArg_
#define occaFunctionInfoArg int occaKernelInfoArg_
#define occaFunctionInfo        occaKernelInfoArg_
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaKernel         extern "C" __global__
#define occaFunction       __device__
#define occaDeviceFunction __device__
//================================================


//---[ Atomics ]----------------------------------
#define occaAtomicAdd(PTR, UPDATE)  atomicAdd
#define occaAtomicSub(PTR, UPDATE)  atomicSub
#define occaAtomicSwap(PTR, UPDATE) atomicExch
#define occaAtomicInc(PTR, UPDATE)  atomicInc
#define occaAtomicDec(PTR, UPDATE)  atomicDec
#define occaAtomicMin(PTR, UPDATE)  atomicMin
#define occaAtomicMax(PTR, UPDATE)  atomicMax
#define occaAtomicAnd(PTR, UPDATE)  atomicAnd
#define occaAtomicOr(PTR, UPDATE)   atomicOr
#define occaAtomicXor(PTR, UPDATE)  atomicXor

#define occaAtomicAdd64  occaAtomicAdd
#define occaAtomicSub64  occaAtomicSub
#define occaAtomicSwap64 occaAtomicSwap
#define occaAtomicInc64  occaAtomicInc
#define occaAtomicDec64  occaAtomicDec
//================================================


//---[ Math ]-------------------------------------
__device__ inline float  occaCuda_fabs(const float x){  return fabsf(x); }
__device__ inline double occaCuda_fabs(const double x){ return fabs(x);  }

#define occaFabs       occaCuda_fabs
#define occaFastFabs   occaCuda_fabs
#define occaNativeFabs occaCuda_fabs

__device__ inline float  occaCuda_sqrt(const float x){      return sqrtf(x);      }
__device__ inline double occaCuda_sqrt(const double x){     return sqrt(x);       }
__device__ inline float  occaCuda_fastSqrt(const float x){  return __fsqrt_rn(x); }
__device__ inline double occaCuda_fastSqrt(const double x){ return __dsqrt_rn(x); }

#define occaSqrt       occaCuda_sqrt
#define occaFastSqrt   occaCuda_fastSqrt
#define occaNativeSqrt occaSqrt

__device__ inline float  occaCuda_cbrt(const float x){  return cbrtf(x); }
__device__ inline double occaCuda_cbrt(const double x){ return cbrt(x);  }

#define occaCbrt       occaCuda_cbrt
#define occaFastCbrt   occaCuda_cbrt
#define occaNativeCbrt occaCuda_cbrt

__device__ inline float  occaCuda_sin(const float x){      return sinf(x);   }
__device__ inline double occaCuda_sin(const double x){     return sin(x);    }
__device__ inline float  occaCuda_fastSin(const float x){  return __sinf(x); }
__device__ inline double occaCuda_fastSin(const double x){ return sin(x);    }

#define occaSin       occaCuda_sin
#define occaFastSin   occaCuda_fastSin
#define occaNativeSin occaSin

__device__ inline float  occaCuda_asin(const float x){  return asinf(x); }
__device__ inline double occaCuda_asin(const double x){ return asin(x);  }

#define occaAsin       occaCuda_asin
#define occaFastAsin   occaCuda_asin
#define occaNativeAsin occaCuda_asin

__device__ inline float  occaCuda_sinh(const float x){  return sinhf(x); }
__device__ inline double occaCuda_sinh(const double x){ return sinh(x);  }

#define occaSinh       occaCuda_sinh
#define occaFastSinh   occaCuda_sinh
#define occaNativeSinh occaCuda_sinh

__device__ inline float  occaCuda_asinh(const float x){  return asinhf(x); }
__device__ inline double occaCuda_asinh(const double x){ return asinh(x);  }

#define occaAsinh       occaCuda_asinh
#define occaFastAsinh   occaCuda_asinh
#define occaNativeAsinh occaCuda_asinh

__device__ inline float  occaCuda_cos(const float x){      return cosf(x);   }
__device__ inline double occaCuda_cos(const double x){     return cos(x);    }
__device__ inline float  occaCuda_fastCos(const float x){  return __cosf(x); }
__device__ inline double occaCuda_fastCos(const double x){ return cos(x);    }

#define occaCos       occaCuda_cos
#define occaFastCos   occaCuda_fastCos
#define occaNativeCos occaCos

__device__ inline float  occaCuda_acos(const float x){ return acosf(x); }
__device__ inline double occaCuda_acos(const double x){ return acos(x); }

#define occaAcos       occaCuda_acos
#define occaFastAcos   occaCuda_acos
#define occaNativeAcos occaCuda_acos

__device__ inline float  occaCuda_cosh(const float x){  return coshf(x); }
__device__ inline double occaCuda_cosh(const double x){ return cosh(x);  }

#define occaCosh       occaCuda_cosh
#define occaFastCosh   occaCuda_cosh
#define occaNativeCosh occaCuda_cosh

__device__ inline float  occaCuda_acosh(const float x){ return acoshf(x); }
__device__ inline double occaCuda_acosh(const double x){ return acosh(x); }

#define occaAcosh       occaCuda_acosh
#define occaFastAcosh   occaCuda_acosh
#define occaNativeAcosh occaCuda_acosh

__device__ inline float  occaCuda_tan(const float x){      return tanf(x);   }
__device__ inline double occaCuda_tan(const double x){     return tan(x);    }
__device__ inline float  occaCuda_fastTan(const float x){  return __tanf(x); }
__device__ inline double occaCuda_fastTan(const double x){ return tan(x);    }

#define occaTan       occaCuda_tan
#define occaFastTan   occaCuda_fastTan
#define occaNativeTan occaTan

__device__ inline float  occaCuda_atan(const float x){  return atanf(x); }
__device__ inline double occaCuda_atan(const double x){ return atan(x);  }

#define occaAtan       occaCuda_atan
#define occaFastAtan   occaCuda_atan
#define occaNativeAtan occaCuda_atan

__device__ inline float  occaCuda_tanh(const float x){  return tanhf(x); }
__device__ inline double occaCuda_tanh(const double x){ return tanh(x);  }

#define occaTanh       occaCuda_tanh
#define occaFastTanh   occaCuda_tanh
#define occaNativeTanh occaCuda_tanh

__device__ inline float  occaCuda_atanh(const float x){  return atanhf(x); }
__device__ inline double occaCuda_atanh(const double x){ return atanh(x);  }

#define occaAtanh       occaCuda_atanh
#define occaFastAtanh   occaCuda_atanh
#define occaNativeAtanh occaCuda_atanh

__device__ inline float  occaCuda_exp(const float x){      return expf(x);   }
__device__ inline double occaCuda_exp(const double x){     return exp(x);    }
__device__ inline float  occaCuda_fastExp(const float x){  return __expf(x); }
__device__ inline double occaCuda_fastExp(const double x){ return exp(x);    }

#define occaExp       occaCuda_exp
#define occaFastExp   occaCuda_fastExp
#define occaNativeExp occaExp

__device__ inline float  occaCuda_expm1(const float x){  return expm1f(x); }
__device__ inline double occaCuda_expm1(const double x){ return expm1(x);  }

#define occaExpm1       occaCuda_expm1
#define occaFastExpm1   occaCuda_expm1
#define occaNativeExpm1 occaCuda_expm1

__device__ inline float  occaCuda_pow(const float x, const float p){      return powf(x,p);   }
__device__ inline double occaCuda_pow(const double x, const double p){     return pow(x,p);    }
__device__ inline float  occaCuda_fastPow(const float x, const float p){  return __powf(x,p); }
__device__ inline double occaCuda_fastPow(const double x, const double p){ return pow(x,p);    }

#define occaPow       occaCuda_pow
#define occaFastPow   occaCuda_fastPow
#define occaNativePow occaPow

__device__ inline float  occaCuda_log2(const float x){      return log2f(x);   }
__device__ inline double occaCuda_log2(    const double x){ return log2(x);    }
__device__ inline float  occaCuda_fastLog2(const float x){  return __log2f(x); }
__device__ inline double occaCuda_fastLog2(const double x){ return log2(x);    }

#define occaLog2       occaCuda_log2
#define occaFastLog2   occaCuda_fastLog2
#define occaNativeLog2 occaLog2

__device__ inline float  occaCuda_log10(const float x){      return log10f(x);   }
__device__ inline double occaCuda_log10(const double x){     return log10(x);    }
__device__ inline float  occaCuda_fastLog10(const float x){  return __log10f(x); }
__device__ inline double occaCuda_fastLog10(const double x){ return log10(x);    }

#define occaLog10       occaCuda_log10
#define occaFastLog10   occaCuda_fastLog10
#define occaNativeLog10 occaLog10
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
#define occaPrivateArray( TYPE , NAME , SIZE ) TYPE NAME[SIZE]
#define occaPrivate( TYPE , NAME )             TYPE NAME
//================================================


//---[ Texture ]----------------------------------
#define occaReadOnly  const
#define occaWriteOnly

#define occaSampler(TEX) __occa__##TEX##__sampler__

#define occaTexture1D(TEX) cudaSurfaceObject_t TEX, int occaSampler(TEX)
#define occaTexture2D(TEX) cudaSurfaceObject_t TEX, int occaSampler(TEX)

#define occaTexGet1D(TEX, TYPE, VALUE, X)    surf1Dread(&(VALUE), TEX, X*sizeof(TYPE)   , (cudaSurfaceBoundaryMode) occaSampler(TEX))
#define occaTexGet2D(TEX, TYPE, VALUE, X, Y) surf2Dread(&(VALUE), TEX, X*sizeof(TYPE), Y, (cudaSurfaceBoundaryMode) occaSampler(TEX))

#define occaTexSet1D(TEX, TYPE, VALUE, X)    surf1Dwrite(VALUE, TEX, X*sizeof(TYPE)   , (cudaSurfaceBoundaryMode) occaSampler(TEX))
#define occaTexSet2D(TEX, TYPE, VALUE, X, Y) surf2Dwrite(VALUE, TEX, X*sizeof(TYPE), Y, (cudaSurfaceBoundaryMode) occaSampler(TEX))
//================================================

#endif
