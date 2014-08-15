#ifndef OCCA_CUDA_DEFINES_HEADER
#define OCCA_CUDA_DEFINES_HEADER

//---[ Defines ]----------------------------------
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

#define occaBarrier(FENCE) __syncthreads()
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaContinue return
//================================================


//---[ Attributes ]-------------------------------
#define occaShared   __shared__
#define occaPointer
#define occaVariable
#define occaRestrict __restrict__
#define occaVolatile __volatile__
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


//---[ Math ]-------------------------------------
inline float  occaCuda_sqrt(const float x){      return sqrtf(x);      }
inline double occaCuda_sqrt(const double x){     return sqrt(x);       }
inline float  occaCuda_fastSqrt(const float x){  return __fsqrt_rn(x); }
inline double occaCuda_fastSqrt(const double x){ return __dsqrt_rn(x); }

#define occaSqrt       occaCuda_sqrt
#define occaFastSqrt   occaCuda_fastSqrt
#define occaNativeSqrt occaSqrt

inline float  occaCuda_sin(const float x){      return sinf(x);   }
inline double occaCuda_sin(const double x){     return sin(x);    }
inline float  occaCuda_fastSin(const float x){  return __sinf(x); }
inline double occaCuda_fastSin(const double x){ return sin(x);    }

#define occaSin       occaCuda_sin
#define occaFastSin   occaCuda_fastSin
#define occaNativeSin occaSin

inline float  occaCuda_asin(const float x){      return asinf(x); }
inline double occaCuda_asin(const double x){     return asin(x);  }
inline float  occaCuda_fastAsin(const float x){  return asinf(x); }
inline double occaCuda_fastAsin(const double x){ return asin(x);  }

#define occaAsin       occaCuda_asin
#define occaFastAsin   occaCuda_fastAsin
#define occaNativeAsin occaAsin

inline float  occaCuda_sinh(const float x){      return sinhf(x); }
inline double occaCuda_sinh(const double x){     return sinh(x);  }
inline float  occaCuda_fastSinh(const float x){  return sinhf(x); }
inline double occaCuda_fastSinh(const double x){ return sinh(x);  }

#define occaSinh       occaCuda_sinh
#define occaFastSinh   occaCuda_fastSinh
#define occaNativeSinh occaSinh

inline float  occaCuda_asinh(const float x){       return asinhf(x); }
inline double occaCuda_asinh(const double x){      return asinh(x);  }
inline float  occaCuda_fastAsinh(const float x){   return asinhf(x); }
inline double occaCuda_fastAsinh(const double x){  return asinh(x);  }

#define occaAsinh       occaCuda_asinh
#define occaFastAsinh   occaCuda_fastAsinh
#define occaNativeAsinh occaAsinh

inline float  occaCuda_cos(const float x){      return cosf(x);   }
inline double occaCuda_cos(const double x){     return cos(x);    }
inline float  occaCuda_fastCos(const float x){  return __cosf(x); }
inline double occaCuda_fastCos(const double x){ return cos(x);    }

#define occaCos       occaCuda_cos
#define occaFastCos   occaCuda_fastCos
#define occaNativeCos occaCos

inline float  occaCuda_acos(const float x){      return acosf(x); }
inline double occaCuda_acos(    const double x){ return acos(x);  }
inline float  occaCuda_fastAcos(const float x){  return acosf(x); }
inline double occaCuda_fastAcos(const double x){ return acos(x);  }

#define occaAcos       occaCuda_acos
#define occaFastAcos   occaCuda_fastAcos
#define occaNativeAcos occaAcos

inline float  occaCuda_cosh(const float x){      return coshf(x); }
inline double occaCuda_cosh(    const double x){ return cosh(x);  }
inline float  occaCuda_fastCosh(const float x){  return coshf(x); }
inline double occaCuda_fastCosh(const double x){ return cosh(x);  }

#define occaCosh       occaCuda_cosh
#define occaFastCosh   occaCuda_fastCosh
#define occaNativeCosh occaCosh

inline float  occaCuda_acosh(const float x){      return acoshf(x); }
inline double occaCuda_acosh    (const double x){ return acosh(x);  }
inline float  occaCuda_fastAcosh(const float x){  return acoshf(x); }
inline double occaCuda_fastAcosh(const double x){ return acosh(x);  }

#define occaAcosh       occaCuda_acosh
#define occaFastAcosh   occaCuda_fastAcosh
#define occaNativeAcosh occaAcosh

inline float  occaCuda_tan(const float x){      return tanf(x);   }
inline double occaCuda_tan(const double x){     return tan(x);    }
inline float  occaCuda_fastTan(const float x){  return __tanf(x); }
inline double occaCuda_fastTan(const double x){ return tan(x);    }

#define occaTan       occaCuda_tan
#define occaFastTan   occaCuda_fastTan
#define occaNativeTan occaTan

inline float  occaCuda_atan(const float x){      return atanf(x); }
inline double occaCuda_atan(const double x){     return atan(x);  }
inline float  occaCuda_fastAtan(const float x){  return atanf(x); }
inline double occaCuda_fastAtan(const double x){ return atan(x);  }

#define occaAtan       occaCuda_atan
#define occaFastAtan   occaCuda_fastAtan
#define occaNativeAtan occaAtan

inline float  occaCuda_tanh(const float x){      return tanhf(x); }
inline double occaCuda_tanh(    const double x){ return tanh(x);  }
inline float  occaCuda_fastTanh(const float x){  return tanhf(x); }
inline double occaCuda_fastTanh(const double x){ return tanh(x);  }

#define occaTanh       occaCuda_tanh
#define occaFastTanh   occaCuda_fastTanh
#define occaNativeTanh occaTanh

inline float  occaCuda_atanh(const float x){      return atanhf(x); }
inline double occaCuda_atanh(const double x){     return atanh(x);  }
inline float  occaCuda_fastAtanh(const float x){  return atanhf(x); }
inline double occaCuda_fastAtanh(const double x){ return atanh(x);  }

#define occaAtanh       occaCuda_atanh
#define occaFastAtanh   occaCuda_fastAtanh
#define occaNativeAtanh occaAtanh

inline float  occaCuda_exp(const float x){      return expf(x);   }
inline double occaCuda_exp(const double x){     return exp(x);    }
inline float  occaCuda_fastExp(const float x){  return __expf(x); }
inline double occaCuda_fastExp(const double x){ return exp(x);    }

#define occaExp       occaCuda_exp
#define occaFastExp   occaCuda_fastExp
#define occaNativeExp occaExp

inline float  occaCuda_log2(const float x){      return log2f(x);   }
inline double occaCuda_log2(    const double x){ return log2(x);    }
inline float  occaCuda_fastLog2(const float x){  return __log2f(x); }
inline double occaCuda_fastLog2(const double x){ return log2(x);    }

#define occaLog2       occaCuda_log2
#define occaFastLog2   occaCuda_fastLog2
#define occaNativeLog2 occaLog2

inline float  occaCuda_log10(const float x){      return log10f(x);   }
inline double occaCuda_log10(const double x){     return log10(x);    }
inline float  occaCuda_fastLog10(const float x){  return __log10f(x); }
inline double occaCuda_fastLog10(const double x){ return log10(x);    }

#define occaLog10       occaCuda_log10
#define occaFastLog10   occaCuda_fastLog10
#define occaNativeLog10 occaLog10
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
