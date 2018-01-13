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

#ifndef OCCA_OPENCL_DEFINES_HEADER
#define OCCA_OPENCL_DEFINES_HEADER

//---[ Defines ]----------------------------------
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define OCCA_IN_KERNEL      1

#define OCCA_USING_SERIAL   0
#define OCCA_USING_OPENMP   0
#define OCCA_USING_OPENCL   1
#define OCCA_USING_CUDA     0
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
#define occaOuterDim2 (get_num_groups(2))
#define occaOuterId2  (get_group_id(2))

#define occaOuterDim1 (get_num_groups(1))
#define occaOuterId1  (get_group_id(1))

#define occaOuterDim0 (get_num_groups(0))
#define occaOuterId0  (get_group_id(0))
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaInnerDim2 (get_local_size(2))
#define occaInnerId2  (get_local_id(2))

#define occaInnerDim1 (get_local_size(1))
#define occaInnerId1  (get_local_id(1))

#define occaInnerDim0 (get_local_size(0))
#define occaInnerId0  (get_local_id(0))
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaGlobalDim2 (get_global_size(2))
#define occaGlobalId2  (get_global_id(2))

#define occaGlobalDim1 (get_global_size(1))
#define occaGlobalId1  (get_global_id(1))

#define occaGlobalDim0 (get_global_size(0))
#define occaGlobalId0  (get_global_id(0))
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
#define occaLocalMemFence CLK_LOCAL_MEM_FENCE
#define occaGlobalMemFence CLK_GLOBAL_MEM_FENCE

#define occaBarrier(FENCE)      barrier(FENCE)
#define occaInnerBarrier(FENCE) barrier(FENCE)
#define occaOuterBarrier(FENCE)
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaContinue return
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaDirectLoad(X) (*(X))
//================================================


//---[ Attributes ]-------------------------------
#define occaShared   __local
#define occaPointer  __global
#define occaVariable
#define occaRestrict restrict
#define occaVolatile volatile
#define occaAligned
#define occaFunctionShared __local
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaConst    const
#define occaConstant __constant
//================================================


//---[ Kernel Info ]------------------------------
#define occaKernelInfoArg   __global void *occaKernelInfoArg_
#define occaFunctionInfoArg __global void *occaKernelInfoArg_
#define occaFunctionInfo                   occaKernelInfoArg_
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaKernel         __kernel
#define occaFunction
#define occaDeviceFunction
//================================================


//---[ Atomics ]----------------------------------
#define occaAtomicAdd(PTR, UPDATE)
#define occaAtomicSub(PTR, UPDATE)
#define occaAtomicSwap(PTR, UPDATE)
#define occaAtomicInc(PTR, UPDATE)
#define occaAtomicDec(PTR, UPDATE)
#define occaAtomicMin(PTR, UPDATE)
#define occaAtomicMax(PTR, UPDATE)
#define occaAtomicAnd(PTR, UPDATE)
#define occaAtomicOr(PTR, UPDATE)
#define occaAtomicXor(PTR, UPDATE)

#define occaAtomicAddL  occaAtomicAdd
#define occaAtomicSubL  occaAtomicSub
#define occaAtomicSwapL occaAtomicSwap
#define occaAtomicIncL  occaAtomicInc
#define occaAtomicDecL  occaAtomicDec
//================================================


//---[ Atomics ]----------------------------------
#ifdef cl_khr_int32_base_atomics
#  pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable
#  define occaAtomicAdd(PTR, UPDATE)        atomic_add(PTR, UPDATE)
#  define occaAtomicSub(PTR, UPDATE)        atomic_sub(PTR, UPDATE)
#  define occaAtomicSwap(PTR, UPDATE)       atomic_xchg(PTR, UPDATE)
#  define occaAtomicInc(PTR, UPDATE)        atomic_inc(PTR, UPDATE)
#  define occaAtomicDec(PTR, UPDATE)        atomic_dec(PTR, UPDATE)
#  define occaAtomicCAS(PTR, COMP, UPDATE) atomic_cmpxchg(PTR, COMP, UPDATE)
#endif

#ifdef cl_khr_int32_extended_atomics
#  pragma OPENCL EXTENSION cl_khr_int32_extended_atomics : enable
#  define occaAtomicMin(PTR, UPDATE)  atomic_min(PTR, UPDATE)
#  define occaAtomicMax(PTR, UPDATE)  atomic_max(PTR, UPDATE)
#  define occaAtomicAnd(PTR, UPDATE)  atomic_and(PTR, UPDATE)
#  define occaAtomicOr(PTR, UPDATE)   atomic_or(PTR, UPDATE)
#  define occaAtomicXor(PTR, UPDATE)  atomic_xor(PTR, UPDATE)
#endif

#ifdef cl_khr_int64_atomics
#  pragma OPENCL EXTENSION cl_khr_int64_atomics : enable
#  define occaAtomicAdd64(PTR, UPDATE)  atom_add(PTR, UPDATE)
#  define occaAtomicSub64(PTR, UPDATE)  atom_sub(PTR, UPDATE)
#  define occaAtomicSwap64(PTR, UPDATE) atom_swap(PTR, UPDATE)
#  define occaAtomicInc64(PTR, UPDATE)  atom_inc(PTR, UPDATE)
#  define occaAtomicDec64(PTR, UPDATE)  atom_dec(PTR, UPDATE)
#endif
//================================================


//---[ Math ]-------------------------------------
#define occaMin(a,b)  (((a) < (b)) ? (a) : (b))
#define occaFastMin   occaMin
#define occaNativeMin occaMin

#define occaMax(a,b)  (((a) > (b)) ? (a) : (b))
#define occaFastMax   occaMax
#define occaNativeMax occaMax

#define occaHypot       hypot
#define occaFastHypot   hypot
#define occaNativeHypot hypot

#define occaFabs       fabs
#define occaFastFabs   fabs
#define occaNativeFabs fabs

#define occaSqrt       sqrt
#define occaFastSqrt   half_sqrt
#define occaNativeSqrt native_sqrt

#define occaCbrt       cbrt
#define occaFastCbrt   cbrt
#define occaNativeCbrt cbrt

#define occaSin       sin
#define occaFastSin   half_sin
#define occaNativeSin native_sin

#define occaAsin       asin
#define occaFastAsin   half_asin
#define occaNativeAsin native_asin

#define occaSinh       sinh
#define occaFastSinh   half_sinh
#define occaNativeSinh native_sinh

#define occaAsinh       asinh
#define occaFastAsinh   half_asinh
#define occaNativeAsinh native_asinh

#define occaCos       cos
#define occaFastCos   half_cos
#define occaNativeCos native_cos

#define occaAcos       acos
#define occaFastAcos   half_acos
#define occaNativeAcos native_acos

#define occaCosh       cosh
#define occaFastCosh   half_cosh
#define occaNativeCosh native_cosh

#define occaAcosh       acosh
#define occaFastAcosh   half_acosh
#define occaNativeAcosh native_acosh

#define occaTan       tan
#define occaFastTan   half_tan
#define occaNativeTan native_tan

#define occaAtan       atan
#define occaFastAtan   half_atan
#define occaNativeAtan native_atan

#define occaAtan2       atan2
#define occaFastAtan2   atan2
#define occaNativeAtan2 atan2

#define occaTanh       tanh
#define occaFastTanh   half_tanh
#define occaNativeTanh native_tanh

#define occaAtanh       atanh
#define occaFastAtanh   half_atanh
#define occaNativeAtanh native_atanh

#define occaExp       exp
#define occaFastExp   half_exp
#define occaNativeExp native_exp

#define occaExpm1       expm1
#define occaFastExpm1   expm1
#define occaNativeExpm1 expm1

#define occaPow       pow
#define occaFastPow   half_pow
#define occaNativePow native_pow

#define occaLog2       log2
#define occaFastLog2   half_log2
#define occaNativeLog2 native_log2

#define occaLog10       log10
#define occaFastLog10   half_log10
#define occaNativeLog10 native_log10
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
#define occaReadOnly  read_only
#define occaWriteOnly write_only

#define occaSampler(TEX) __occa__##TEX##__sampler__

#if __OPENCL_VERSION__ < 120
#  define occaTexture1D(TEX) __global void *TEX
#else
#  define occaTexture1D(TEX) image1d_t TEX, occaConst sampler_t occaSampler(TEX)
#endif

#define occaTexture2D(TEX) image2d_t TEX, occaConst sampler_t occaSampler(TEX)

#define occaTexGet1D_int(TEX, X)    read_imagei(TEX, occaSampler(TEX), (int2) {X, 1}).x
#define occaTexGet2D_int(TEX, X, Y) read_imagei(TEX, occaSampler(TEX), (int2) {X, Y}).x

#define occaTexGet1D_int2(TEX, X)    read_imagei(TEX, occaSampler(TEX), (int2) {X, 1}).xy
#define occaTexGet2D_int2(TEX, X, Y) read_imagei(TEX, occaSampler(TEX), (int2) {X, Y}).xy

#define occaTexGet1D_int4(TEX, X)    read_imagei(TEX, occaSampler(TEX), (int2) {X, 1})
#define occaTexGet2D_int4(TEX, X, Y) read_imagei(TEX, occaSampler(TEX), (int2) {X, Y})

#define occaTexGet1D_uint(TEX, X)    read_imageui(TEX, occaSampler(TEX), (int2) {X, 1}).x
#define occaTexGet2D_uint(TEX, X, Y) read_imageui(TEX, occaSampler(TEX), (int2) {X, Y}).x

#define occaTexGet1D_uint2(TEX, X)    read_imageui(TEX, occaSampler(TEX), (int2) {X, 1}).xy
#define occaTexGet2D_uint2(TEX, X, Y) read_imageui(TEX, occaSampler(TEX), (int2) {X, Y}).xy

#define occaTexGet1D_uint4(TEX, X)    read_imageui(TEX, occaSampler(TEX), (int2) {X, 1})
#define occaTexGet2D_uint4(TEX, X, Y) read_imageui(TEX, occaSampler(TEX), (int2) {X, Y})

#define occaTexGet1D_float(TEX, X)    read_imagef(TEX, occaSampler(TEX), (int2) {X, 1}).x
#define occaTexGet2D_float(TEX, X, Y) read_imagef(TEX, occaSampler(TEX), (int2) {X, Y}).x

#define occaTexGet1D_float2(TEX, X)    read_imagef(TEX, occaSampler(TEX), (int2) {X, 1}).xy
#define occaTexGet2D_float2(TEX, X, Y) read_imagef(TEX, occaSampler(TEX), (int2) {X, Y}).xy

#define occaTexGet1D_float4(TEX, X)    read_imagef(TEX, occaSampler(TEX), (int2) {X, 1})
#define occaTexGet2D_float4(TEX, X, Y) read_imagef(TEX, occaSampler(TEX), (int2) {X, Y})

#define occaTexSet1D_int(TEX, VALUE, X)    write_imagei(TEX,  X           , (int4) {VALUE, 0, 0, 0})
#define occaTexSet2D_int(TEX, VALUE, X, Y) write_imagei(TEX, (int2) {X, Y}, (int4) {VALUE, 0, 0, 0})

#define occaTexSet1D_int2(TEX, VALUE, X)    write_imagei(TEX,  X           , (int4) {VALUE.x, VALUE.y, 0, 0})
#define occaTexSet2D_int2(TEX, VALUE, X, Y) write_imagei(TEX, (int2) {X, Y}, (int4) {VALUE.x, VALUE.y, 0, 0})

#define occaTexSet1D_int4(TEX, VALUE, X)    write_imagei(TEX,  X           , VALUE)
#define occaTexSet2D_int4(TEX, VALUE, X, Y) write_imagei(TEX, (int2) {X, Y}, VALUE)

#define occaTexSet1D_uint(TEX, VALUE, X)    write_imageui(TEX,  X           , (uint4) {VALUE, 0, 0, 0})
#define occaTexSet2D_uint(TEX, VALUE, X, Y) write_imageui(TEX, (int2) {X, Y}, (uint4) {VALUE, 0, 0, 0})

#define occaTexSet1D_uint2(TEX, VALUE, X)    write_imageui(TEX,  X           , (uint4) {VALUE.x, VALUE.y, 0, 0})
#define occaTexSet2D_uint2(TEX, VALUE, X, Y) write_imageui(TEX, (int2) {X, Y}, (uint4) {VALUE.x, VALUE.y, 0, 0})

#define occaTexSet1D_uint4(TEX, VALUE, X)    write_imageui(TEX,  X           , VALUE)
#define occaTexSet2D_uint4(TEX, VALUE, X, Y) write_imageui(TEX, (int2) {X, Y}, VALUE)

#define occaTexSet1D_float(TEX, VALUE, X)    write_imagef(TEX, X            , (float4) {VALUE, 0, 0, 0})
#define occaTexSet2D_float(TEX, VALUE, X, Y) write_imagef(TEX, (int2) {X, Y}, (float4) {VALUE, 0, 0, 0})

#define occaTexSet1D_float2(TEX, VALUE, X)    write_imagef(TEX,  X           , (float4) {VALUE.x, VALUE.y, 0, 0})
#define occaTexSet2D_float2(TEX, VALUE, X, Y) write_imagef(TEX, (int2) {X, Y}, (float4) {VALUE.x, VALUE.y, 0, 0})

#define occaTexSet1D_float4(TEX, VALUE, X)    write_imagef(TEX,  X           , VALUE)
#define occaTexSet2D_float4(TEX, VALUE, X, Y) write_imagef(TEX, (int2) {X, Y}, VALUE)

#if __OPENCL_VERSION__ < 120
#  define occaTexGet1D(TEX, TYPE, VALUE, X) VALUE = ((TYPE*) TEX)[X]
#  define occaTexSet1D(TEX, TYPE, VALUE, X) ((TYPE*) TEX)[X] = VALUE
#else
#  define occaTexGet1D(TEX, TYPE, VALUE, X) VALUE = occaTexGet1D_##TYPE(TEX, X)
#  define occaTexSet1D(TEX, TYPE, VALUE, X) occaTexSet1D_##TYPE(TEX, VALUE, X)
#endif

#define occaTexGet2D(TEX, TYPE, VALUE, X, Y) VALUE = occaTexGet2D_##TYPE(TEX, X, Y)
#define occaTexSet2D(TEX, TYPE, VALUE, X, Y) occaTexSet2D_##TYPE(TEX, VALUE, X, Y)
//================================================

#endif
