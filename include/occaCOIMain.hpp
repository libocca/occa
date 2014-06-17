// COINATIVELIBEXPORT
// void occaKernelWith4Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
//                                void *hostArgs_, uint16_t hostArgSize,
//                                void *ret, uint16_t retSize){
//   char *hostArgs = (char*) hostArgs_;
//   int hostPos = 0;

//   occaKernelWith4Arguments_t kernel = (occaKernelWith4Arguments_t) *((void**) hostArgs);
//   hostPos += sizeof(void*);

//   occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
//   hostPos += 6*sizeof(int);

//   int *types = (int*) (hostArgs + hostPos);

//   int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

//   void *argv[4];

//   for(int i = 0; i < 4; ++i){
//     if(types[i] & (1 << 31))
//       argv[i] = deviceArgs[ types[i] - (1 << 31) ];
//     else
//       argv[i] = (void*) (hostArgs + types[i]);
//   }

//   kernel(&kernelArgs, occaInnerId0, occaInnerId2, occaInnerId2,
//          argv[0] , argv[1] , argv[2] , argv[3] );
// }


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>

#include <omp.h>

#include <intel-coi/sink/COIPipeline_sink.h>
#include <intel-coi/sink/COIProcess_sink.h>
#include <intel-coi/sink/COIBuffer_sink.h>
#include <intel-coi/common/COIMacros_common.h>

struct occaKernelArgs_t {
  int outerZ, outerY, outerX;
  int innerZ, innerY, innerX;
};

int main(int argc, char **argv){
  UNUSED_ATTR COIRESULT started = COIPipelineStartExecutingRunFunctions();

  if(started != COI_SUCCESS){
    printf("COI Kernel failed upon launch.\n");
    return 0;
  }

  COIProcessWaitForShutdown();

  return 0;
}

typedef void(*occaKernelWith1Argument_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 );

typedef void(*occaKernelWith2Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                          void *arg0 , void *arg1 );

typedef void(*occaKernelWith3Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                          void *arg0 , void *arg1 , void *arg2 );

typedef void(*occaKernelWith4Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                          void *arg0 , void *arg1 , void *arg2 , void *arg3 );

typedef void(*occaKernelWith5Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                          void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 );


typedef void(*occaKernelWith6Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                          void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                          void *arg5 );

typedef void(*occaKernelWith7Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                          void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                          void *arg5 , void *arg6 );

typedef void(*occaKernelWith8Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                          void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                          void *arg5 , void *arg6 , void *arg7 );

typedef void(*occaKernelWith9Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                          void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                          void *arg5 , void *arg6 , void *arg7 , void *arg8 );

typedef void(*occaKernelWith10Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                           void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                           void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 );


typedef void(*occaKernelWith11Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                           void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                           void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                           void *arg10);

typedef void(*occaKernelWith12Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                           void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                           void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                           void *arg10, void *arg11);

typedef void(*occaKernelWith13Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                           void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                           void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                           void *arg10, void *arg11, void *arg12);

typedef void(*occaKernelWith14Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                           void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                           void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                           void *arg10, void *arg11, void *arg12, void *arg13);

typedef void(*occaKernelWith15Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                           void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                           void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                           void *arg10, void *arg11, void *arg12, void *arg13, void *arg14);


typedef void(*occaKernelWith16Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                           void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                           void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                           void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                           void *arg15);

typedef void(*occaKernelWith17Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                           void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                           void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                           void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                           void *arg15, void *arg16);

typedef void(*occaKernelWith18Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                           void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                           void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                           void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                           void *arg15, void *arg16, void *arg17);

typedef void(*occaKernelWith19Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                           void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                           void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                           void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                           void *arg15, void *arg16, void *arg17, void *arg18);

typedef void(*occaKernelWith20Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                           void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                           void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                           void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                           void *arg15, void *arg16, void *arg17, void *arg18, void *arg19);

typedef void(*occaKernelWith21Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                           void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                           void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                           void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                           void *arg15, void *arg16, void *arg17, void *arg18, void *arg19,
                                           void *arg20);

typedef void(*occaKernelWith22Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                           void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                           void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                           void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                           void *arg15, void *arg16, void *arg17, void *arg18, void *arg19,
                                           void *arg20, void *arg21);

typedef void(*occaKernelWith23Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                           void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                           void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                           void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                           void *arg15, void *arg16, void *arg17, void *arg18, void *arg19,
                                           void *arg20, void *arg21, void *arg22);

typedef void(*occaKernelWith24Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                           void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                           void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                           void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                           void *arg15, void *arg16, void *arg17, void *arg18, void *arg19,
                                           void *arg20, void *arg21, void *arg22, void *arg23);

typedef void(*occaKernelWith25Arguments_t)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                           void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                           void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                           void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                           void *arg15, void *arg16, void *arg17, void *arg18, void *arg19,
                                           void *arg20, void *arg21, void *arg22, void *arg23, void *arg24);

1,2,3,4,5,6,7,8,9,10,
1,2,3,4,5,6,7,8,9,10,
21,2,3,4,5,

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] );
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments_t kernel = (occaKernelWith25Arguments_t) *((void**) hostArgs);
  hostPos += sizeof(void*);

  occaKernelArgs_t kernelArgs = *((occaKernelArgs_t*) (hostArgs + hostPos));
  hostPos += 6*sizeof(int);

  int *types = (int*) (hostArgs + hostPos);

  int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 31))
      argv[i] = deviceArgs[ types[i] - (1 << 31) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(&&kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8] , argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}
