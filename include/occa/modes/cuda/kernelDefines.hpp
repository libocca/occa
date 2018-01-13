/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
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

#ifndef OCCA_CUDA_DEFINES_HEADER
#define OCCA_CUDA_DEFINES_HEADER

//---[ Defines ]----------------------------------
#define OCCA_IN_KERNEL      1

#define OCCA_USING_SERIAL   0
#define OCCA_USING_OPENMP   0
#define OCCA_USING_OPENCL   0
#define OCCA_USING_CUDA     1
#define OCCA_USING_PTHREADS 0

#define OCCA_USING_CPU (OCCA_USING_SERIAL || OCCA_USING_OPENMP || OCCA_USING_PTHREADS)
#define OCCA_USING_GPU (OCCA_USING_OPENCL || OCCA_USING_CUDA)
//================================================

//---[ Remove CPU Defines ]-----------------------
//  |---[ Compiler ]----------
#define OCCA_GNU_COMPILER       (1 << 0)
#define OCCA_LLVM_COMPILER      (1 << 1)
#define OCCA_INTEL_COMPILER     (1 << 2)
#define OCCA_PATHSCALE_COMPILER (1 << 3)
#define OCCA_IBM_COMPILER       (1 << 4)
#define OCCA_PGI_COMPILER       (1 << 5)
#define OCCA_HP_COMPILER        (1 << 6)
#define OCCA_VS_COMPILER        (1 << 7)
#define OCCA_CRAY_COMPILER      (1 << 8)
#define OCCA_UNKNOWN_COMPILER   (1 << 9)

#define OCCA_COMPILED_WITH OCCA_UNKNOWN_COMPILER

//  |---[ Vectorization ]-----
#define OCCA_MIC    0
#define OCCA_AVX2   0
#define OCCA_AVX    0
#define OCCA_SSE4_2 0
#define OCCA_SSE4_1 0
#define OCCA_SSE4   0
#define OCCA_SSE3   0
#define OCCA_SSE2   0
#define OCCA_SSE    0
#define OCCA_MMX    0

#define OCCA_VECTOR_SET

#define OCCA_SIMD_WIDTH 0
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
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaDirectLoad(X) (__ldg(X))
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
#define occaAtomicAdd(PTR, UPDATE)       atomicAdd(PTR, UPDATE)
#define occaAtomicSub(PTR, UPDATE)       atomicSub(PTR, UPDATE)
#define occaAtomicSwap(PTR, UPDATE)      atomicExch(PTR, UPDATE)
#define occaAtomicInc(PTR, UPDATE)       atomicInc(PTR, UPDATE)
#define occaAtomicDec(PTR, UPDATE)       atomicDec(PTR, UPDATE)
#define occaAtomicMin(PTR, UPDATE)       atomicMin(PTR, UPDATE)
#define occaAtomicMax(PTR, UPDATE)       atomicMax(PTR, UPDATE)
#define occaAtomicAnd(PTR, UPDATE)       atomicAnd(PTR, UPDATE)
#define occaAtomicOr(PTR, UPDATE)        atomicOr(PTR, UPDATE)
#define occaAtomicXor(PTR, UPDATE)       atomicXor(PTR, UPDATE)
#define occaAtomicCAS(PTR, COMP, UPDATE) atomicCAS(PTR, COMP, UPDATE)

#define occaAtomicAdd64  occaAtomicAdd
#define occaAtomicSub64  occaAtomicSub
#define occaAtomicSwap64 occaAtomicSwap
#define occaAtomicInc64  occaAtomicInc
#define occaAtomicDec64  occaAtomicDec
//================================================


//---[ Math ]-------------------------------------
template <class TM>
__device__ inline TM occaCuda_min(const TM a, const TM b) { return (((a) < (b)) ? (a) : (b)); }

#define occaMin       occaCuda_min
#define occaFastMin   occaCuda_min
#define occaNativeMin occaCuda_min

template <class TM>
__device__ inline TM occaCuda_max(const TM a, const TM b) { return (((a) > (b)) ? (a) : (b)); }

#define occaMax       occaCuda_max
#define occaFastMax   occaCuda_max
#define occaNativeMax occaCuda_max

__device__ inline float  occaCuda_hypot(const float x,  const float  y) {  return hypotf(x,y); }
__device__ inline double occaCuda_hypot(const double x, const double y) {  return hypot(x,y);  }

#define occaHypot       occaCuda_hypot
#define occaFastHypot   occaCuda_hypot
#define occaNativeHypot occaCuda_hypot

template <class TM>
__device__ inline TM     occaCuda_fabs(const TM x)    {  return x >= 0 ? x : -x; }
template <>
__device__ inline float  occaCuda_fabs(const float x) {  return fabsf(x); }
template <>
__device__ inline double occaCuda_fabs(const double x) { return fabs(x);  }

#define occaFabs       occaCuda_fabs
#define occaFastFabs   occaCuda_fabs
#define occaNativeFabs occaCuda_fabs

template <class TM>
__device__ inline TM     occaCuda_sqrt(const TM x)         { return sqrt(x);       }
template <>
__device__ inline float  occaCuda_sqrt(const float x)      { return sqrtf(x);      }
template <>
__device__ inline double occaCuda_sqrt(const double x)     { return sqrt(x);       }

template <class TM>
__device__ inline TM     occaCuda_fastSqrt(const TM x)     { return sqrt(x);       }
template <>
__device__ inline float  occaCuda_fastSqrt(const float x)  { return __fsqrt_rn(x); }
template <>
__device__ inline double occaCuda_fastSqrt(const double x) { return __dsqrt_rn(x); }

#define occaSqrt       occaCuda_sqrt
#define occaFastSqrt   occaCuda_fastSqrt
#define occaNativeSqrt occaSqrt

__device__ inline float  occaCuda_cbrt(const float x) {  return cbrtf(x); }
__device__ inline double occaCuda_cbrt(const double x) { return cbrt(x);  }

#define occaCbrt       occaCuda_cbrt
#define occaFastCbrt   occaCuda_cbrt
#define occaNativeCbrt occaCuda_cbrt

__device__ inline float  occaCuda_sin(const float x) {      return sinf(x);   }
__device__ inline double occaCuda_sin(const double x) {     return sin(x);    }
__device__ inline float  occaCuda_fastSin(const float x) {  return __sinf(x); }
__device__ inline double occaCuda_fastSin(const double x) { return sin(x);    }

#define occaSin       occaCuda_sin
#define occaFastSin   occaCuda_fastSin
#define occaNativeSin occaSin

__device__ inline float  occaCuda_asin(const float x) {  return asinf(x); }
__device__ inline double occaCuda_asin(const double x) { return asin(x);  }

#define occaAsin       occaCuda_asin
#define occaFastAsin   occaCuda_asin
#define occaNativeAsin occaCuda_asin

__device__ inline float  occaCuda_sinh(const float x) {  return sinhf(x); }
__device__ inline double occaCuda_sinh(const double x) { return sinh(x);  }

#define occaSinh       occaCuda_sinh
#define occaFastSinh   occaCuda_sinh
#define occaNativeSinh occaCuda_sinh

__device__ inline float  occaCuda_asinh(const float x) {  return asinhf(x); }
__device__ inline double occaCuda_asinh(const double x) { return asinh(x);  }

#define occaAsinh       occaCuda_asinh
#define occaFastAsinh   occaCuda_asinh
#define occaNativeAsinh occaCuda_asinh

__device__ inline float  occaCuda_cos(const float x) {      return cosf(x);   }
__device__ inline double occaCuda_cos(const double x) {     return cos(x);    }
__device__ inline float  occaCuda_fastCos(const float x) {  return __cosf(x); }
__device__ inline double occaCuda_fastCos(const double x) { return cos(x);    }

#define occaCos       occaCuda_cos
#define occaFastCos   occaCuda_fastCos
#define occaNativeCos occaCos

__device__ inline float  occaCuda_acos(const float x) { return acosf(x); }
__device__ inline double occaCuda_acos(const double x) { return acos(x); }

#define occaAcos       occaCuda_acos
#define occaFastAcos   occaCuda_acos
#define occaNativeAcos occaCuda_acos

__device__ inline float  occaCuda_cosh(const float x) {  return coshf(x); }
__device__ inline double occaCuda_cosh(const double x) { return cosh(x);  }

#define occaCosh       occaCuda_cosh
#define occaFastCosh   occaCuda_cosh
#define occaNativeCosh occaCuda_cosh

__device__ inline float  occaCuda_acosh(const float x) { return acoshf(x); }
__device__ inline double occaCuda_acosh(const double x) { return acosh(x); }

#define occaAcosh       occaCuda_acosh
#define occaFastAcosh   occaCuda_acosh
#define occaNativeAcosh occaCuda_acosh

__device__ inline float  occaCuda_tan(const float x) {      return tanf(x);   }
__device__ inline double occaCuda_tan(const double x) {     return tan(x);    }
__device__ inline float  occaCuda_fastTan(const float x) {  return __tanf(x); }
__device__ inline double occaCuda_fastTan(const double x) { return tan(x);    }

#define occaTan       occaCuda_tan
#define occaFastTan   occaCuda_fastTan
#define occaNativeTan occaTan

__device__ inline float  occaCuda_atan(const float x) {  return atanf(x); }
__device__ inline double occaCuda_atan(const double x) { return atan(x);  }

#define occaAtan       occaCuda_atan
#define occaFastAtan   occaCuda_atan
#define occaNativeAtan occaCuda_atan

__device__ inline float  occaCuda_atan2(const float y, const float x) {  return atan2f(y,x); }
__device__ inline double occaCuda_atan2(const double y, const double x) { return atan2(y,x);  }

#define occaAtan2       occaCuda_atan2
#define occaFastAtan2   occaCuda_atan2
#define occaNativeAtan2 occaCuda_atan2

__device__ inline float  occaCuda_tanh(const float x) {  return tanhf(x); }
__device__ inline double occaCuda_tanh(const double x) { return tanh(x);  }

#define occaTanh       occaCuda_tanh
#define occaFastTanh   occaCuda_tanh
#define occaNativeTanh occaCuda_tanh

__device__ inline float  occaCuda_atanh(const float x) {  return atanhf(x); }
__device__ inline double occaCuda_atanh(const double x) { return atanh(x);  }

#define occaAtanh       occaCuda_atanh
#define occaFastAtanh   occaCuda_atanh
#define occaNativeAtanh occaCuda_atanh

__device__ inline float  occaCuda_exp(const float x) {      return expf(x);   }
__device__ inline double occaCuda_exp(const double x) {     return exp(x);    }
__device__ inline float  occaCuda_fastExp(const float x) {  return __expf(x); }
__device__ inline double occaCuda_fastExp(const double x) { return exp(x);    }

#define occaExp       occaCuda_exp
#define occaFastExp   occaCuda_fastExp
#define occaNativeExp occaExp

__device__ inline float  occaCuda_expm1(const float x) {  return expm1f(x); }
__device__ inline double occaCuda_expm1(const double x) { return expm1(x);  }

#define occaExpm1       occaCuda_expm1
#define occaFastExpm1   occaCuda_expm1
#define occaNativeExpm1 occaCuda_expm1

__device__ inline float  occaCuda_pow(const float x, const float p) {      return powf(x,p);   }
__device__ inline double occaCuda_pow(const double x, const double p) {     return pow(x,p);    }
__device__ inline float  occaCuda_fastPow(const float x, const float p) {  return __powf(x,p); }
__device__ inline double occaCuda_fastPow(const double x, const double p) { return pow(x,p);    }

#define occaPow       occaCuda_pow
#define occaFastPow   occaCuda_fastPow
#define occaNativePow occaPow

__device__ inline float  occaCuda_log2(const float x) {      return log2f(x);   }
__device__ inline double occaCuda_log2(    const double x) { return log2(x);    }
__device__ inline float  occaCuda_fastLog2(const float x) {  return __log2f(x); }
__device__ inline double occaCuda_fastLog2(const double x) { return log2(x);    }

#define occaLog2       occaCuda_log2
#define occaFastLog2   occaCuda_fastLog2
#define occaNativeLog2 occaLog2

__device__ inline float  occaCuda_log10(const float x) {      return log10f(x);   }
__device__ inline double occaCuda_log10(const double x) {     return log10(x);    }
__device__ inline float  occaCuda_fastLog10(const float x) {  return __log10f(x); }
__device__ inline double occaCuda_fastLog10(const double x) { return log10(x);    }

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
