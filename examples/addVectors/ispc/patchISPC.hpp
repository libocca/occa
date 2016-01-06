#undef  occaInnerFor
#define occaInnerFor foreach(occaInnerId0 = 0 ... occaInnerDim0, occaInnerId1 = 0 ... occaInnerDim1, occaInnerId2 = 0 ... occaInnerDim2) 

#undef  occaInnerFor0
#define occaInnerFor0 foreach(occaInnerId0 = 0 ... occaInnerDim0)

// not sure what to do with these
#undef occaInnerFor1
#define occaInnerFor1

#undef occaInnerFor2
#define occaInnerFor2

// use occaOuterFor to launch
#undef occaOuterFor
#define occaOuterFor launch [ occaOuterFor0, occaOuterFor1, occaOuterFor2 ]

#undef occaOuterFor0
#define occaOuterFor0 occaOuterFor

#undef occaOuterFor1
#define occaOuterFor1

#undef occaOuterFor2
#define occaOuterFor2

#undef occaParallelFor
#define occaParallelFor

#undef occaKernel
#define occaKernel export

// useful macros
#define occaGlobalId  (occaInnerId0 + occaInnerDim0*(occaInnerId1 + occaInnerDim1*(occaInnerId2 + occaOuterDim0*(occaOuterId0 + occaOuterDim1*(occaOuterId1 + occaOuterDim2*occaOuterId2)))))
#define occaGlobalDim (occaOuterDim0*occaOuterDim1*occaOuterDim2)
