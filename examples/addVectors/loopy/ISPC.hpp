// variable macros
#define occaPointer uniform
#define occaVariable &
#define occaConst const
#define occaRestrict 

// function macros
#define occaFunction task

#define occaFunctionInfoArg const uniform int occaRestrict const occaKernelArgs[] 
#define occaFunctionInfo                             occaKernelArgs

// kernel info macros
#define occaKernelInfoArg   const uniform int occaRestrict const occaKernelArgs[], uniform int occaInnerIdA, uniform int occaInnerIdB, uniform int occaInnerIdC

#define occaKernel export

// outer indices read from ISPC
#define occaOuterId0 taskIndex0
#define occaOuterId1 taskIndex1
#define occaOuterId2 taskIndex2


// parallel for loop macros
#define occaInnerFor foreach(occaInnerId0 = 0 ... occaInnerDim0, occaInnerId1 = 0 ... occaInnerDim1, occaInnerId2 = 0 ... occaInnerDim2) 

#define occaInnerFor0 foreach(occaInnerId0 = 0 ... occaInnerDim0)

#define occaInnerFor1

#define occaInnerFor2

// use occaOuterFor to launch
#define occaOuterFor launch [ occaOuterDim0, occaOuterDim1, occaOuterDim2 ]

#define occaOuterFor0 occaOuterFor

#define occaOuterFor1

#define occaOuterFor2

#define occaParallelFor

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


// useful macros
#define occaGlobalId  (occaInnerId0 + occaInnerDim0*(occaInnerId1 + occaInnerDim1*(occaInnerId2 + occaOuterDim0*(occaOuterId0 + occaOuterDim1*(occaOuterId1 + occaOuterDim2*occaOuterId2)))))
#define occaGlobalDim (occaOuterDim0*occaOuterDim1*occaOuterDim2)
