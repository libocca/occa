#ifndef OCCA_OPENMP_DEFINES_HEADER
#define OCCA_OPENMP_DEFINES_HEADER

#include <cmath>

#if OCCA_OPENMP_ENABLED
#  include "omp.h"
#endif

//---[ Defines ]----------------------------------
#define OCCA_MAX_THREADS 512
#define OCCA_MEM_ALIGN   16

typedef struct occaArgs_t_ { int data[12]; } occaArgs_t;

typedef struct float2_t { float  x,y;     } float2;
typedef struct float3_t { float  x,y,z;   } float3;
typedef struct float4_t { float  x,y,z,w; } float4;

typedef struct double2_t { double  x,y;     } double2;
typedef struct double3_t { double  x,y,z;   } double3;
typedef struct double4_t { double  x,y,z,w; } double4;
//================================================


//---[ Loop Info ]--------------------------------
#define occaOuterDim2 occaArgs.data[0]
#define occaOuterId2  occaArgs.data[1]

#define occaOuterDim1 occaArgs.data[2]
#define occaOuterId1  occaArgs.data[3]

#define occaOuterDim0 occaArgs.data[4]
#define occaOuterId0  occaArgs.data[5]
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaInnerDim2 occaArgs.data[6]
#define occaInnerId2  occaArgs.data[7]

#define occaInnerDim1 occaArgs.data[8]
#define occaInnerId1  occaArgs.data[9]

#define occaInnerDim0 occaArgs.data[10]
#define occaInnerId0  occaArgs.data[11]
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaGlobalDim2 (occaInnerDim2 * occaOuterDim2)
#define occaGlobalId2  (occaOuterId2*occaInnerDim2 + occaInnerId2)

#define occaGlobalDim1 (occaInnerDim1 * occaOuterDim1)
#define occaGlobalId1  (occaOuterId1*occaInnerDim1 + occaInnerId1)

#define occaGlobalDim0 (occaInnerDim0 * occaOuterDim0)
#define occaGlobalId0  (occaOuterId0*occaInnerDim0 + occaInnerId0)
//================================================


//---[ Loops ]------------------------------------
#define occaOuterFor2 for(occaOuterId2 = 0; occaOuterId2 < occaOuterDim2; ++occaOuterId2)
#define occaOuterFor1 for(occaOuterId1 = 0; occaOuterId1 < occaOuterDim1; ++occaOuterId1)

#if OCCA_OPENMP_ENABLED
#  define occaOuterFor0                                                   \
  _Pragma("omp parallel for firstprivate(occaOuterId0, occaInnerId0, occaInnerId1, occaInnerId2)") \
  for(occaOuterId0 = 0; occaOuterId0 < occaOuterDim0; ++occaOuterId0)
#else
#  define occaOuterFor0                                         \
  for(occaOuterId0 = 0; occaOuterId0 < occaOuterDim0; ++occaOuterId0)
#endif
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
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaContinue continue
//================================================


//---[ Attributes ]-------------------------------
#define occaShared
#define occaPointer
#define occaVariable &
#define occaRestrict __restrict__
#define occaVolatile volatile
#define occaAligned  __attribute__ ((aligned (OCCA_MEM_ALIGN)))
#define occaFunctionShared
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaConst    const
#define occaConstant const
//================================================


//---[ Kernel Info ]------------------------------
#define occaKernelInfoArg   occaArgs_t &occaArgs
#define occaFunctionInfoArg occaArgs_t &occaArgs
#define occaFunctionInfo              occaArgs
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaKernel         extern "C"
#define occaFunction
#define occaDeviceFunction
//================================================


//---[ Private ]---------------------------------
template <class TM, const int SIZE>
class occaPrivate_t {
private:

public:
  const occaArgs_t &occaArgs;

  TM data[OCCA_MAX_THREADS][SIZE] occaAligned;

  occaPrivate_t(occaArgs_t &occaArgs_) :
    occaArgs(occaArgs_) {}

  ~occaPrivate_t(){}

#define OCCA_PRIVATE_ID                                        \
  (occaInnerId2*occaInnerDim1 + occaInnerId1)*occaInnerDim0 + occaInnerId0

  inline TM& operator [] (const int n){
    return data[OCCA_PRIVATE_ID][n];
  }

  inline operator TM(){
    return data[OCCA_PRIVATE_ID][0];
  }

  inline occaPrivate_t& operator = (const occaPrivate_t &r) {
    const int id = OCCA_PRIVATE_ID;
    data[id][0] = r.data[id][0];
  }

  inline occaPrivate_t<TM,SIZE> & operator = (const TM &t){
    data[OCCA_PRIVATE_ID][0] = t;
    return *this;
  }

  inline occaPrivate_t<TM,SIZE> & operator += (const TM &t){
    data[OCCA_PRIVATE_ID][0] += t;
    return *this;
  }

  inline occaPrivate_t<TM,SIZE> & operator -= (const TM &t){
    data[OCCA_PRIVATE_ID][0] -= t;
    return *this;
  }

  inline occaPrivate_t<TM,SIZE> & operator /= (const TM &t){
    data[OCCA_PRIVATE_ID][0] /= t;
    return *this;
  }

  inline occaPrivate_t<TM,SIZE> & operator *= (const TM &t){
    data[OCCA_PRIVATE_ID][0] *= t;
    return *this;
  }
};

#define occaPrivateArray( TYPE , NAME , SIZE )   \
  occaPrivate_t<TYPE,SIZE> NAME(occaArgs);

#define occaPrivate( TYPE , NAME )               \
  occaPrivate_t<TYPE,1> NAME(occaArgs);
//================================================

#endif
