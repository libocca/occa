#ifndef OCCA_OPENCL_DEFINES_HEADER
#define OCCA_OPENCL_DEFINES_HEADER

//---[ Defines ]----------------------------------
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
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

#define occaBarrier(FENCE) barrier(FENCE)
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaContinue return
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


//---[ Math ]-------------------------------------
#define occaSqrt       sqrt
#define occaFastSqrt   half_sqrt
#define occaNativeSqrt native_sqrt

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

#define occaTanh       tanh
#define occaFastTanh   half_tanh
#define occaNativeTanh native_tanh

#define occaAtanh       atanh
#define occaFastAtanh   half_atanh
#define occaNativeAtanh native_atanh

#define occaExp       exp
#define occaFastExp   half_exp
#define occaNativeExp native_exp

#define occaLog2       log2
#define occaFastLog2   half_log2
#define occaNativeLog2 native_log2

#define occaLog10       log10
#define occaFastLog10   half_log10
#define occaNativeLog10 native_log10
//================================================


//---[ Misc ]-------------------------------------
#define occaUnroll3(N) _Pragma(#N)
#define occaUnroll2(N) occaUnroll3(N)
#define occaUnroll(N)  occaUnroll2(unroll N)
//================================================


//---[ Private ]---------------------------------
#define occaPrivateArray( TYPE , NAME , SIZE ) TYPE NAME[SIZE]
#define occaPrivate( TYPE , NAME )             TYPE NAME
//================================================

#endif
