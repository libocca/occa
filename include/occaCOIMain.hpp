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

typedef void(*occaKernelWith1Argument)(void *arg0 );

typedef void(*occaKernelWith2Arguments)(void *arg0 , void *arg1 );

typedef void(*occaKernelWith3Arguments)(void *arg0 , void *arg1 , void *arg2 );

typedef void(*occaKernelWith4Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 );

typedef void(*occaKernelWith5Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 );


typedef void(*occaKernelWith6Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                        void *arg5 );

typedef void(*occaKernelWith7Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                        void *arg5 , void *arg6 );

typedef void(*occaKernelWith8Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                        void *arg5 , void *arg6 , void *arg7 );

typedef void(*occaKernelWith9Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                        void *arg5 , void *arg6 , void *arg7 , void *arg8 );

typedef void(*occaKernelWith10Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 );


typedef void(*occaKernelWith11Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10);

typedef void(*occaKernelWith12Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11);

typedef void(*occaKernelWith13Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12);

typedef void(*occaKernelWith14Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13);

typedef void(*occaKernelWith15Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14);


typedef void(*occaKernelWith16Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15);

typedef void(*occaKernelWith17Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16);

typedef void(*occaKernelWith18Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16, void *arg17);

typedef void(*occaKernelWith19Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16, void *arg17, void *arg18);

typedef void(*occaKernelWith20Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16, void *arg17, void *arg18, void *arg19);

typedef void(*occaKernelWith21Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16, void *arg17, void *arg18, void *arg19,
                                         void *arg20);

typedef void(*occaKernelWith22Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16, void *arg17, void *arg18, void *arg19,
                                         void *arg20, void *arg21);

typedef void(*occaKernelWith23Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16, void *arg17, void *arg18, void *arg19,
                                         void *arg20, void *arg21, void *arg22);

typedef void(*occaKernelWith24Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16, void *arg17, void *arg18, void *arg19,
                                         void *arg20, void *arg21, void *arg22, void *arg23);

typedef void(*occaKernelWith25Arguments)(void *arg0 , void *arg1 , void *arg2 , void *arg3 , void *arg4 ,
                                         void *arg5 , void *arg6 , void *arg7 , void *arg8 , void *arg9 ,
                                         void *arg10, void *arg11, void *arg12, void *arg13, void *arg14,
                                         void *arg15, void *arg16, void *arg17, void *arg18, void *arg19,
                                         void *arg20, void *arg21, void *arg22, void *arg23, void *arg24);

COINATIVELIBEXPORT
void occaKernelWith1Argument(uint32_t argc, void **argv, uint64_t *argSize,
                             void *miscArg, uint16_t miscArgSize,
                             void *ret, uint16_t retSize){
  occaKernelWith1Argument kernel = (occaKernelWith1Argument) argv[0];

  kernel(argv[1]);
}

COINATIVELIBEXPORT
void occaKernelWith2Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                              void *miscArg, uint16_t miscArgSize,
                              void *ret, uint16_t retSize){
  occaKernelWith2Arguments kernel = (occaKernelWith2Arguments) argv[0];

  kernel(argv[1], argv[2]);
}

COINATIVELIBEXPORT
void occaKernelWith3Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                              void *miscArg, uint16_t miscArgSize,
                              void *ret, uint16_t retSize){
  occaKernelWith3Arguments kernel = (occaKernelWith3Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3]);
}

COINATIVELIBEXPORT
void occaKernelWith4Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                              void *miscArg, uint16_t miscArgSize,
                              void *ret, uint16_t retSize){
  occaKernelWith4Arguments kernel = (occaKernelWith4Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4]);
}

COINATIVELIBEXPORT
void occaKernelWith5Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                              void *miscArg, uint16_t miscArgSize,
                              void *ret, uint16_t retSize){
  occaKernelWith5Arguments kernel = (occaKernelWith5Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5]);
}

COINATIVELIBEXPORT
void occaKernelWith6Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                              void *miscArg, uint16_t miscArgSize,
                              void *ret, uint16_t retSize){
  occaKernelWith6Arguments kernel = (occaKernelWith6Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6]);
}

COINATIVELIBEXPORT
void occaKernelWith7Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                              void *miscArg, uint16_t miscArgSize,
                              void *ret, uint16_t retSize){
  occaKernelWith7Arguments kernel = (occaKernelWith7Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7]);
}

COINATIVELIBEXPORT
void occaKernelWith8Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                              void *miscArg, uint16_t miscArgSize,
                              void *ret, uint16_t retSize){
  occaKernelWith8Arguments kernel = (occaKernelWith8Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8]);
}

COINATIVELIBEXPORT
void occaKernelWith9Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                              void *miscArg, uint16_t miscArgSize,
                              void *ret, uint16_t retSize){
  occaKernelWith9Arguments kernel = (occaKernelWith9Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9]);
}

COINATIVELIBEXPORT
void occaKernelWith10Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                               void *miscArg, uint16_t miscArgSize,
                               void *ret, uint16_t retSize){
  occaKernelWith10Arguments kernel = (occaKernelWith10Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9] , argv[10]);
}

COINATIVELIBEXPORT
void occaKernelWith11Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                               void *miscArg, uint16_t miscArgSize,
                               void *ret, uint16_t retSize){
  occaKernelWith11Arguments kernel = (occaKernelWith11Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9] , argv[10],
         argv[11]);
}

COINATIVELIBEXPORT
void occaKernelWith12Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                               void *miscArg, uint16_t miscArgSize,
                               void *ret, uint16_t retSize){
  occaKernelWith12Arguments kernel = (occaKernelWith12Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9] , argv[10],
         argv[11], argv[12]);
}

COINATIVELIBEXPORT
void occaKernelWith13Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                               void *miscArg, uint16_t miscArgSize,
                               void *ret, uint16_t retSize){
  occaKernelWith13Arguments kernel = (occaKernelWith13Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9] , argv[10],
         argv[11], argv[12], argv[13]);
}

COINATIVELIBEXPORT
void occaKernelWith14Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                               void *miscArg, uint16_t miscArgSize,
                               void *ret, uint16_t retSize){
  occaKernelWith14Arguments kernel = (occaKernelWith14Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9] , argv[10],
         argv[11], argv[12], argv[13], argv[14]);
}

COINATIVELIBEXPORT
void occaKernelWith15Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                               void *miscArg, uint16_t miscArgSize,
                               void *ret, uint16_t retSize){
  occaKernelWith15Arguments kernel = (occaKernelWith15Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9] , argv[10],
         argv[11], argv[12], argv[13], argv[14], argv[15]);
}

COINATIVELIBEXPORT
void occaKernelWith16Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                               void *miscArg, uint16_t miscArgSize,
                               void *ret, uint16_t retSize){
  occaKernelWith16Arguments kernel = (occaKernelWith16Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9] , argv[10],
         argv[11], argv[12], argv[13], argv[14], argv[15],
         argv[16]);
}

COINATIVELIBEXPORT
void occaKernelWith17Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                               void *miscArg, uint16_t miscArgSize,
                               void *ret, uint16_t retSize){
  occaKernelWith17Arguments kernel = (occaKernelWith17Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9] , argv[10],
         argv[11], argv[12], argv[13], argv[14], argv[15],
         argv[16], argv[17]);
}

COINATIVELIBEXPORT
void occaKernelWith18Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                               void *miscArg, uint16_t miscArgSize,
                               void *ret, uint16_t retSize){
  occaKernelWith18Arguments kernel = (occaKernelWith18Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9] , argv[10],
         argv[11], argv[12], argv[13], argv[14], argv[15],
         argv[16], argv[17], argv[18]);
}

COINATIVELIBEXPORT
void occaKernelWith19Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                               void *miscArg, uint16_t miscArgSize,
                               void *ret, uint16_t retSize){
  occaKernelWith19Arguments kernel = (occaKernelWith19Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9] , argv[10],
         argv[11], argv[12], argv[13], argv[14], argv[15],
         argv[16], argv[17], argv[18], argv[19]);
}

COINATIVELIBEXPORT
void occaKernelWith20Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                               void *miscArg, uint16_t miscArgSize,
                               void *ret, uint16_t retSize){
  occaKernelWith20Arguments kernel = (occaKernelWith20Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9] , argv[10],
         argv[11], argv[12], argv[13], argv[14], argv[15],
         argv[16], argv[17], argv[18], argv[19], argv[20]);
}

COINATIVELIBEXPORT
void occaKernelWith21Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                               void *miscArg, uint16_t miscArgSize,
                               void *ret, uint16_t retSize){
  occaKernelWith21Arguments kernel = (occaKernelWith21Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9] , argv[10],
         argv[11], argv[12], argv[13], argv[14], argv[15],
         argv[16], argv[17], argv[18], argv[19], argv[20],
         argv[21]);
}

COINATIVELIBEXPORT
void occaKernelWith22Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                               void *miscArg, uint16_t miscArgSize,
                               void *ret, uint16_t retSize){
  occaKernelWith22Arguments kernel = (occaKernelWith22Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9] , argv[10],
         argv[11], argv[12], argv[13], argv[14], argv[15],
         argv[16], argv[17], argv[18], argv[19], argv[20],
         argv[21], argv[22]);
}

COINATIVELIBEXPORT
void occaKernelWith23Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                               void *miscArg, uint16_t miscArgSize,
                               void *ret, uint16_t retSize){
  occaKernelWith23Arguments kernel = (occaKernelWith23Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9] , argv[10],
         argv[11], argv[12], argv[13], argv[14], argv[15],
         argv[16], argv[17], argv[18], argv[19], argv[20],
         argv[21], argv[22], argv[23]);
}

COINATIVELIBEXPORT
void occaKernelWith24Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                               void *miscArg, uint16_t miscArgSize,
                               void *ret, uint16_t retSize){
  occaKernelWith24Arguments kernel = (occaKernelWith24Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9] , argv[10],
         argv[11], argv[12], argv[13], argv[14], argv[15],
         argv[16], argv[17], argv[18], argv[19], argv[20],
         argv[21], argv[22], argv[23], argv[24]);
}

COINATIVELIBEXPORT
void occaKernelWith25Arguments(uint32_t argc, void **argv, uint64_t *argSize,
                               void *miscArg, uint16_t miscArgSize,
                               void *ret, uint16_t retSize){
  occaKernelWith25Arguments kernel = (occaKernelWith25Arguments) argv[0];

  kernel(argv[1] , argv[2] , argv[3] , argv[4] , argv[5] ,
         argv[6] , argv[7] , argv[8] , argv[9] , argv[10],
         argv[11], argv[12], argv[13], argv[14], argv[15],
         argv[16], argv[17], argv[18], argv[19], argv[20],
         argv[21], argv[22], argv[23], argv[24], argv[25]);
}
