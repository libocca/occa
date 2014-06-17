#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>

#include <omp.h>

#include <intel-coi/sink/COIPipeline_sink.h>
#include <intel-coi/sink/COIProcess_sink.h>
#include <intel-coi/sink/COIBuffer_sink.h>
#include <intel-coi/common/COIMacros_common.h>

int main(int argc, char **argv){
  UNUSED_ATTR COIRESULT started = COIPipelineStartExecutingRunFunctions();

  if(started != COI_SUCCESS){
    printf("COI Kernel failed upon launch.\n");
    return 0;
  }

  COIProcessWaitForShutdown();

  return 0;
}

typedef void(*occaKernelWith1Argument)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                       void *arg0 );

typedef void(*occaKernelWith2Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                        void *arg0 , void *arg1 );

typedef void(*occaKernelWith3Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                        void *arg0 , void *arg1 , void *arg2 );

typedef void(*occaKernelWith4Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                        void *arg0 , void *arg1 , void *arg2 , void *arg3 );

typedef void(*occaKernelWith5Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                        void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 );


typedef void(*occaKernelWith6Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                        void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                        void *arg5 );

typedef void(*occaKernelWith7Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                        void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                        void *arg5 , void *arg6 );

typedef void(*occaKernelWith8Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                        void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                        void *arg5 , void *arg6 , void *arg7 );

typedef void(*occaKernelWith9Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                        void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                        void *arg5 , void *arg6 , void *arg7 , void *arg8 );

typedef void(*occaKernelWith10Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 );


typedef void(*occaKernelWith11Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10);

typedef void(*occaKernelWith12Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11);

typedef void(*occaKernelWith13Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12);

typedef void(*occaKernelWith14Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13);

typedef void(*occaKernelWith15Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14);


typedef void(*occaKernelWith16Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15);

typedef void(*occaKernelWith17Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16);

typedef void(*occaKernelWith18Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16, void *arg17);

typedef void(*occaKernelWith19Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16, void *arg17, void *arg18);

typedef void(*occaKernelWith20Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16, void *arg17, void *arg18, void *arg19);

typedef void(*occaKernelWith21Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16, void *arg17, void *arg18, void *arg19,
                                         void *arg20);

typedef void(*occaKernelWith22Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16, void *arg17, void *arg18, void *arg19,
                                         void *arg20, void *arg21);

typedef void(*occaKernelWith23Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16, void *arg17, void *arg18, void *arg19,
                                         void *arg20, void *arg21, void *arg22);

typedef void(*occaKernelWith24Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16, void *arg17, void *arg18, void *arg19,
                                         void *arg20, void *arg21, void *arg22, void *arg23);

typedef void(*occaKernelWith25Arguments)(void *kernelArgs, int occaInnerId0, int occaInnerId1, int occaInnerId2,
                                         void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16, void *arg17, void *arg18, void *arg19,
                                         void *arg20, void *arg21, void *arg22, void *arg23, void *arg24);

COINATIVELIBEXPORT
void occaKernelWith1Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith1Arguments kernel = (occaKernelWith1Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[1];

  for(int i = 0; i < 1; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] );
}

COINATIVELIBEXPORT
void occaKernelWith2Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith2Arguments kernel = (occaKernelWith2Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[2];

  for(int i = 0; i < 2; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] );
}

COINATIVELIBEXPORT
void occaKernelWith3Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith3Arguments kernel = (occaKernelWith3Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[3];

  for(int i = 0; i < 3; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] );
}

COINATIVELIBEXPORT
void occaKernelWith4Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith4Arguments kernel = (occaKernelWith4Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[4];

  for(int i = 0; i < 4; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] );
}

COINATIVELIBEXPORT
void occaKernelWith5Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith5Arguments kernel = (occaKernelWith5Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[5];

  for(int i = 0; i < 5; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] );
}

COINATIVELIBEXPORT
void occaKernelWith6Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith6Arguments kernel = (occaKernelWith6Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[6];

  for(int i = 0; i < 6; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] );
}

COINATIVELIBEXPORT
void occaKernelWith7Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith7Arguments kernel = (occaKernelWith7Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[7];

  for(int i = 0; i < 7; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] );
}

COINATIVELIBEXPORT
void occaKernelWith8Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith8Arguments kernel = (occaKernelWith8Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[8];

  for(int i = 0; i < 8; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] );
}

COINATIVELIBEXPORT
void occaKernelWith9Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith9Arguments kernel = (occaKernelWith9Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[9];

  for(int i = 0; i < 9; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8]);
}

COINATIVELIBEXPORT
void occaKernelWith10Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith10Arguments kernel = (occaKernelWith10Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[10];

  for(int i = 0; i < 10; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8], argv[9]);
}

COINATIVELIBEXPORT
void occaKernelWith11Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith11Arguments kernel = (occaKernelWith11Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[11];

  for(int i = 0; i < 11; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8], argv[9],
         argv[10]);
}

COINATIVELIBEXPORT
void occaKernelWith12Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith12Arguments kernel = (occaKernelWith12Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[12];

  for(int i = 0; i < 12; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8], argv[9],
         argv[10], argv[11]);
}

COINATIVELIBEXPORT
void occaKernelWith13Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith13Arguments kernel = (occaKernelWith13Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[13];

  for(int i = 0; i < 13; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8], argv[9],
         argv[10], argv[11], argv[12]);
}

COINATIVELIBEXPORT
void occaKernelWith14Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith14Arguments kernel = (occaKernelWith14Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[14];

  for(int i = 0; i < 14; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8], argv[9],
         argv[10], argv[11], argv[12], argv[13]);
}

COINATIVELIBEXPORT
void occaKernelWith15Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith15Arguments kernel = (occaKernelWith15Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[15];

  for(int i = 0; i < 15; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8], argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14]);
}

COINATIVELIBEXPORT
void occaKernelWith16Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith16Arguments kernel = (occaKernelWith16Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[16];

  for(int i = 0; i < 16; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8], argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15]);
}

COINATIVELIBEXPORT
void occaKernelWith17Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith17Arguments kernel = (occaKernelWith17Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[17];

  for(int i = 0; i < 17; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8], argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16]);
}

COINATIVELIBEXPORT
void occaKernelWith18Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith18Arguments kernel = (occaKernelWith18Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[18];

  for(int i = 0; i < 18; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8], argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17]);
}

COINATIVELIBEXPORT
void occaKernelWith19Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith19Arguments kernel = (occaKernelWith19Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[19];

  for(int i = 0; i < 19; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8], argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18]);
}

COINATIVELIBEXPORT
void occaKernelWith20Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith20Arguments kernel = (occaKernelWith20Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[20];

  for(int i = 0; i < 20; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8], argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19]);
}

COINATIVELIBEXPORT
void occaKernelWith21Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith21Arguments kernel = (occaKernelWith21Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[21];

  for(int i = 0; i < 21; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8], argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20]);
}

COINATIVELIBEXPORT
void occaKernelWith22Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith22Arguments kernel = (occaKernelWith22Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[22];

  for(int i = 0; i < 22; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8], argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21]);
}

COINATIVELIBEXPORT
void occaKernelWith23Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith23Arguments kernel = (occaKernelWith23Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[23];

  for(int i = 0; i < 23; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8], argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22]);
}

COINATIVELIBEXPORT
void occaKernelWith24Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith24Arguments kernel = (occaKernelWith24Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[24];

  for(int i = 0; i < 24; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8], argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t deviceArgc, void **deviceArgs, uint64_t *deviceArgSize,
                               void *hostArgs_, uint16_t hostArgSize,
                               void *ret, uint16_t retSize){
  char *hostArgs = (char*) hostArgs_;
  int hostPos = 0;

  occaKernelWith25Arguments kernel = (occaKernelWith25Arguments) hostArgs + hostPos;
  hostPos += sizeof(void*);

  void *kernelArgs = hostArgs + hostPos;
  hostPos += 6*sizeof(int);

  char *types = (char*) (hostArgs + hostPos);

  int occaInnerId0, occaInnerId1, occaInnerId2;

  void *argv[25];

  for(int i = 0; i < 25; ++i){
    if(types[i] & (1 << 7))
      argv[i] = deviceArgs[ types[i] & (0 << 7) ];
    else
      argv[i] = (void*) (hostArgs + types[i]);
  }

  kernel(kernelArgs , occaInnerId0, occaInnerId2, occaInnerId2,
         argv[0] , argv[1] , argv[2] , argv[3] , argv[4] ,
         argv[5] , argv[6] , argv[7] , argv[8], argv[9],
         argv[10], argv[11], argv[12], argv[13], argv[14],
         argv[15], argv[16], argv[17], argv[18], argv[19],
         argv[20], argv[21], argv[22], argv[23], argv[24]);
}
