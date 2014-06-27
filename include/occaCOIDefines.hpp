#ifndef OCCA_COI_DEFINES_HEADER
#define OCCA_COI_DEFINES_HEADER
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

typedef struct float2_t { float  x,y;     } float2;
typedef struct float3_t { float  x,y,z;   } float3;
typedef struct float4_t { float  x,y,z,w; } float4;

typedef struct double2_t { double  x,y;     } double2;
typedef struct double3_t { double  x,y,z;   } double3;
typedef struct double4_t { double  x,y,z,w; } double4;
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
#define occaKernelInfoArg   const int *occaKernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2
#define occaFunctionInfoArg const int *occaKernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2
#define occaFunctionInfo               occaKernelArgs,     occaInnerId0,     occaInnerId1,     occaInnerId2
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaKernel         COINATIVELIBEXPORT
#define occaFunction
#define occaDeviceFunction
//================================================


//---[ Misc ]-------------------------------------
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

  inline int index(){
    return ((id2*dim1 + id1)*dim0 + id0);
  }

  inline TM& operator [] (const int n){
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

#endif
