#ifndef OCCA_FBASE_HEADER
#define OCCA_FBASE_HEADER

#include <stdint.h>
#include <stdio.h>
#include "occaCBase.hpp"

/*
 * The code related to fortran string handling was adapted from petsc.  The
 * license information for petsc can be found below.
 *
 *     Licensing Notification
 *
 *       Permission to use, reproduce, prepare derivative works, and to
 *       redistribute to others this software, derivatives of this software, and
 *       future versions of this software as well as its documentation is hereby
 *       granted, provided that this notice is retained thereon and on all
 *       copies or modifications. This permission is perpetual, world-wide, and
 *       provided on a royalty-free basis. UChicago Argonne, LLC and all other
 *       contributors make no representations as to the suitability and
 *       operability of this software for any purpose. It is provided "as is"
 *       without express or implied warranty.
 *
 *      Authors: http://www.mcs.anl.gov/petsc/miscellaneous/index.html
 *
 *
 *      - Mathematics and Computer Science Division
 *      - Argonne National Laboratory
 *      - Argonne IL 60439
 *
 *
 *       Portions of this software are copyright by UChicago Argonne, LLC.
 *       Argonne National Laboratory with facilities in the state of Illinois,
 *       is owned by The United States Government, and operated by UChicago
 *       Argonne, LLC under provision of a contract with the Department of
 *       Energy.
 *
 *     DISCLAIMER
 *
 *       PORTIONS OF THIS SOFTWARE WERE PREPARED AS AN ACCOUNT OF WORK SPONSORED
 *       BY AN AGENCY OF THE UNITED STATES GOVERNMENT. NEITHER THE UNITED STATES
 *       GOVERNMENT NOR ANY AGENCY THEREOF, NOR THE UNIVERSITY OF CHICAGO, NOR
 *       ANY OF THEIR EMPLOYEES OR OFFICERS, MAKES ANY WARRANTY, EXPRESS OR
 *       IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE
 *       ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS,
 *       PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
 *       INFRINGE PRIVATELY OWNED RIGHTS. REFERENCE HEREIN TO ANY SPECIFIC
 *       COMMERCIAL PRODUCT, PROCESS, OR SERVICE BY TRADE NAME, TRADEMARK,
 *       MANUFACTURER, OR OTHERWISE, DOES NOT NECESSARILY CONSTITUTE OR IMPLY
 *       ITS ENDORSEMENT, RECOMMENDATION, OR FAVORING BY THE UNITED STATES
 *       GOVERNMENT OR ANY AGENCY THEREOF. THE VIEW AND OPINIONS OF AUTHORS
 *       EXPRESSED HEREIN DO NOT NECESSARILY STATE OR REFLECT THOSE OF THE
 *       UNITED STATES GOVERNMENT OR ANY AGENCY THEREOF.
 *
 */

#include <stdlib.h>
#include <string.h>

#define OCCA_F2C_NULL_CHARACTER_Fortran ((char *) 0)

/* --------------------------------------------------------------------*/
/*
    This lets us map the str-len argument either, immediately following
    the char argument (DVF on Win32) or at the end of the argument list
    (general unix compilers)
*/
#if defined(OCCAF_HAVE_FORTRAN_MIXED_STR_ARG)
#define OCCA_F2C_LSTR(len) ,int len
#define OCCA_F2C_RSTR(len)
#else
#define OCCA_F2C_LSTR(len)
#define OCCA_F2C_RSTR(len)   ,int len
#endif

/* --------------------------------------------------------------------*/
#define OCCA_F2C_ALLOC_STR(a,n,b) \
do {\
  if (a == OCCA_F2C_NULL_CHARACTER_Fortran) { \
    b = 0; \
  } else { \
    while((n > 0) && (a[n-1] == ' ')) n--; \
    b = (char*)malloc((n+1)*sizeof(char)); \
    if(b==NULL) abort(); \
    strncpy(b,a,n); \
    b[n] = '\0'; \
  } \
} while (0)

#define OCCA_F2C_FREE_STR(a,b) \
do {\
  if (a != b) free(b);\
} while (0)

#define OCCA_F2C_GLOBAL_(name,NAME) name##_

#define  OCCAINT_FC                      OCCA_F2C_GLOBAL_(occaint_fc                    , OCCAINT_FC)
#define  OCCAINT32_FC                    OCCA_F2C_GLOBAL_(occaint32_fc                  , OCCAINT32_FC)
#define  OCCAUINT_FC                     OCCA_F2C_GLOBAL_(occauint_fc                   , OCCAUINT_FC)
#define  OCCACHAR_FC                     OCCA_F2C_GLOBAL_(occachar_fc                   , OCCACHAR_FC)
#define  OCCAUCHAR_FC                    OCCA_F2C_GLOBAL_(occauchar_fc                  , OCCAUCHAR_FC)
#define  OCCASHORT_FC                    OCCA_F2C_GLOBAL_(occashort_fc                  , OCCASHORT_FC)
#define  OCCAUSHORT_FC                   OCCA_F2C_GLOBAL_(occaushort_fc                 , OCCAUSHORT_FC)
#define  OCCALONG_FC                     OCCA_F2C_GLOBAL_(occalong_fc                   , OCCALONG_FC)
#define  OCCAULONG_FC                    OCCA_F2C_GLOBAL_(occaulong_fc                  , OCCAULONG_FC)
#define  OCCAFLOAT_FC                    OCCA_F2C_GLOBAL_(occafloat_fc                  , OCCAFLOAT_FC)
#define  OCCADOUBLE_FC                   OCCA_F2C_GLOBAL_(occadouble_fc                 , OCCADOUBLE_FC)
#define  OCCASTRING_FC                   OCCA_F2C_GLOBAL_(occastring_fc                 , OCCASTRING_FC)
#define  OCCADEVICEMODE_FC               OCCA_F2C_GLOBAL_(occadevicemode_fc             , OCCADEVICEMODE_FC)
#define  OCCADEVICESETCOMPILER_FC        OCCA_F2C_GLOBAL_(occadevicesetcompiler_fc      , OCCADEVICESETCOMPILER_FC)
#define  OCCADEVICESETCOMPILERFLAGS_FC   OCCA_F2C_GLOBAL_(occadevicesetcompilerflags_fc , OCCADEVICESETCOMPILERFLAGS_FC)
#define  OCCAGETDEVICE_FC                OCCA_F2C_GLOBAL_(occagetdevice_fc              , OCCAGETDEVICE_FC)
#define  OCCABUILDKERNELFROMSOURCE_FC    OCCA_F2C_GLOBAL_(occabuildkernelfromsource_fc  , OCCABUILDKERNELFROMSOURCE_FC)
#define  OCCABUILDKERNELFROMBINARY_FC    OCCA_F2C_GLOBAL_(occabuildkernelfrombinary_fc  , OCCABUILDKERNELFROMBINARY_FC)
#define  OCCABUILDKERNELFROMLOOPY_FC     OCCA_F2C_GLOBAL_(occabuildkernelfromloopy_fc   , OCCABUILDKERNELFROMLOOPY_FC)
#define  OCCADEVICEMALLOCNULL_FC         OCCA_F2C_GLOBAL_(occadevicemallocnull_fc       , OCCADEVICEMALLOCNULL_FC)
#define  OCCADEVICEMALLOC_FC             OCCA_F2C_GLOBAL_(occadevicemalloc_fc           , OCCADEVICEMALLOC_FC)
#define  OCCADEVICEFLUSH_FC              OCCA_F2C_GLOBAL_(occadeviceflush_fc            , OCCADEVICEFLUSH_FC)
#define  OCCADEVICEFINISH_FC             OCCA_F2C_GLOBAL_(occadevicefinish_fc           , OCCADEVICEFINISH_FC)
#define  OCCADEVICEGENSTREAM_FC          OCCA_F2C_GLOBAL_(occadevicegenstream_fc        , OCCADEVICEGENSTREAM_FC)
#define  OCCADEVICEGETSTREAM_FC          OCCA_F2C_GLOBAL_(occadevicegetstream_fc        , OCCADEVICEGETSTREAM_FC)
#define  OCCADEVICESETSTREAM_FC          OCCA_F2C_GLOBAL_(occadevicesetstream_fc        , OCCADEVICESETSTREAM_FC)
#define  OCCADEVICETAGSTREAM_FC          OCCA_F2C_GLOBAL_(occadevicetagstream_fc        , OCCADEVICETAGSTREAM_FC)
#define  OCCADEVICETIMEBETWEENTAGS_FC    OCCA_F2C_GLOBAL_(occadevicetimebetweentags_fc  , OCCADEVICETIMEBETWEENTAGS_FC)
#define  OCCADEVICESTREAMFREE_FC         OCCA_F2C_GLOBAL_(occadevicestreamfree_fc       , OCCADEVICESTREAMFREE_FC)
#define  OCCADEVICEFREE_FC               OCCA_F2C_GLOBAL_(occadevicefree_fc             , OCCADEVICEFREE_FC)
#define  OCCAKERNELMODE_FC               OCCA_F2C_GLOBAL_(occakernelmode_fc             , OCCAKERNELMODE_FC)
#define  OCCAKERNELPREFERREDDIMSIZE_FC   OCCA_F2C_GLOBAL_(occakernelpreferreddimsize_fc , OCCAKERNELPREFERREDDIMSIZE_FC)
#define  OCCAKERNELSETWORKINGDIMS_FC     OCCA_F2C_GLOBAL_(occakernelsetworkingdims_fc   , OCCAKERNELSETWORKINGDIMS_FC)
#define  OCCAKERNELSETALLWORKINGDIMS_FC  OCCA_F2C_GLOBAL_(occakernelsetallworkingdims_fc, OCCAKERNELSETALLWORKINGDIMS_FC)
#define  OCCAKERNELTIMETAKEN_FC          OCCA_F2C_GLOBAL_(occakerneltimetaken_fc        , OCCAKERNELTIMETAKEN_FC)
#define  OCCAGENARGUMENTLIST_FC          OCCA_F2C_GLOBAL_(occagenargumentlist_fc        , OCCAGENARGUMENTLIST_FC)
#define  OCCAARGUMENTLISTCLEAR_FC        OCCA_F2C_GLOBAL_(occaargumentlistclear_fc      , OCCAARGUMENTLISTCLEAR_FC)
#define  OCCAARGUMENTLISTFREE_FC         OCCA_F2C_GLOBAL_(occaargumentlistfree_fc       , OCCAARGUMENTLISTFREE_FC)
#define  OCCAARGUMENTLISTADDARGMEM_FC    OCCA_F2C_GLOBAL_(occaargumentlistaddargmem_fc  , OCCAARGUMENTLISTADDARGMEM_FC)
#define  OCCAARGUMENTLISTADDARGTYPE_FC   OCCA_F2C_GLOBAL_(occaargumentlistaddargtype_fc , OCCAARGUMENTLISTADDARGTYPE_FC)
#define  OCCAKERNELRUN__FC               OCCA_F2C_GLOBAL_(occakernelrun__fc             , OCCAKERNELRUN__FC)
#define  OCCAKERNELFREE_FC               OCCA_F2C_GLOBAL_(occakernelfree_fc             , OCCAKERNELFREE_FC)
#define  OCCAGENKERNELINFO_FC            OCCA_F2C_GLOBAL_(occagenkernelinfo_fc          , OCCAGENKERNELINFO_FC)
#define  OCCAKERNELINFOADDDEFINE_FC      OCCA_F2C_GLOBAL_(occakernelinfoadddefine_fc    , OCCAKERNELINFOADDDEFINE_FC)
#define  OCCAKERNELINFOFREE_FC           OCCA_F2C_GLOBAL_(occakernelinfofree_fc         , OCCAKERNELINFOFREE_FC)
#define  OCCAMEMORYMODE_FC               OCCA_F2C_GLOBAL_(occamemorymode_fc             , OCCAMEMORYMODE_FC)
#define  OCCACOPYMEMTOMEM_FC             OCCA_F2C_GLOBAL_(occacopymemtomem_fc           , OCCACOPYMEMTOMEM_FC)
#define  OCCACOPYPTRTOMEM_FC             OCCA_F2C_GLOBAL_(occacopyptrtomem_fc           , OCCACOPYPTRTOMEM_FC)
#define  OCCACOPYMEMTOPTR_FC             OCCA_F2C_GLOBAL_(occacopymemtoptr_fc           , OCCACOPYMEMTOPTR_FC)
#define  OCCAASYNCCOPYMEMTOMEM_FC        OCCA_F2C_GLOBAL_(occaasynccopymemtomem_fc      , OCCAASYNCCOPYMEMTOMEM_FC)
#define  OCCAASYNCCOPYPTRTOMEM_FC        OCCA_F2C_GLOBAL_(occaasynccopyptrtomem_fc      , OCCAASYNCCOPYPTRTOMEM_FC)
#define  OCCAASYNCCOPYMEMTOPTR_FC        OCCA_F2C_GLOBAL_(occaasynccopymemtoptr_fc      , OCCAASYNCCOPYMEMTOPTR_FC)
#define  OCCAMEMORYSWAP_FC               OCCA_F2C_GLOBAL_(occamemoryswap_fc             , OCCAMEMORYSWAP_FC)
#define  OCCAMEMORYFREE_FC               OCCA_F2C_GLOBAL_(occamemoryfree_fc             , OCCAMEMORYFREE_FC)


#  ifdef __cplusplus
extern "C" {
#  endif


  //---[ TypeCasting ]------------------
  void OCCAINT32_FC(occaType *type, int32_t *value){
    if(sizeof(int) == 4)
      *type = occaInt(*value);
    else {
      fprintf(stderr, "Bad integer size\n");
      throw 1;
    }
  }

  void OCCAINT_FC(occaType *type, int *value){
    *type = occaInt(*value);
  }

  void OCCAUINT_FC(occaType *type, unsigned int *value){
    *type = occaUInt(*value);
  }

  void OCCACHAR_FC(occaType *type, char *value){
    *type = occaChar(*value);
  }
  void OCCAUCHAR_FC(occaType *type, unsigned char *value){
    *type = occaUChar(*value);
  }

  void OCCASHORT_FC(occaType *type, short *value){
    *type = occaShort(*value);
  }
  void OCCAUSHORT_FC(occaType *type, unsigned short *value){
    *type = occaUShort(*value);
  }

  void OCCALONG_FC(occaType *type, long *value){
    *type = occaLong(*value);
  }
  void OCCAULONG_FC(occaType *type, unsigned long *value){
    *type = occaULong(*value);
  }

  void OCCAFLOAT_FC(occaType *type, float *value){
    *type = occaFloat(*value);
  }
  void OCCADOUBLE_FC(occaType *type, double *value){
    *type = occaDouble(*value);
  }

  void OCCASTRING_FC(occaType *type, char *str OCCA_F2C_LSTR(str_l)
                     OCCA_F2C_RSTR(str_l)){
    char *str_c;
    OCCA_F2C_ALLOC_STR(str, str_l, str_c);

    *type = occaString(str_c);

    OCCA_F2C_FREE_STR(str, str_c);
  }
  //====================================


  //---[ Device ]-----------------------
  const char* OCCADEVICEMODE_FC(occaDevice device){ // [-]
    return occaDeviceMode(device);
  }

  void OCCADEVICESETCOMPILER_FC(occaDevice *device,
                                const char *compiler OCCA_F2C_LSTR(compiler_l)
                                OCCA_F2C_RSTR(compiler_l)){
    char *compiler_c;
    OCCA_F2C_ALLOC_STR(compiler, compiler_l, compiler_c);

    occaDeviceSetCompiler(*device, compiler_c);

    OCCA_F2C_FREE_STR(compiler, compiler_c);
  }

  void OCCADEVICESETCOMPILERFLAGS_FC(occaDevice *device,
                                     const char *compilerFlags OCCA_F2C_LSTR(compilerFlags_l)
                                     OCCA_F2C_RSTR(compilerFlags_l)){
    char *compilerFlags_c;
    OCCA_F2C_ALLOC_STR(compilerFlags, compilerFlags_l, compilerFlags_c);

    occaDeviceSetCompilerFlags(*device, compilerFlags_c);

    OCCA_F2C_FREE_STR(compilerFlags, compilerFlags_c);
  }

  void OCCAGETDEVICE_FC(occaDevice *device, const char *mode OCCA_F2C_LSTR(mode_l),
                        int32_t *arg1, int32_t *arg2 OCCA_F2C_RSTR(mode_l)){
    char *mode_c;
    OCCA_F2C_ALLOC_STR(mode, mode_l, mode_c);

    *device = occaGetDevice(mode_c, *arg1, *arg2);

    OCCA_F2C_FREE_STR(mode, mode_c);
  }

  void OCCABUILDKERNELFROMSOURCE_FC(occaKernel *kernel, occaDevice *device,
                                    const char *filename     OCCA_F2C_LSTR(filename_l),
                                    const char *functionName OCCA_F2C_LSTR(functionName_l),
                                    occaKernelInfo *info
                                    OCCA_F2C_RSTR(filename_l)
                                    OCCA_F2C_RSTR(functionName_l)){
    char *filename_c, *functionName_c;
    OCCA_F2C_ALLOC_STR(filename    , filename_l    , filename_c);
    OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

    *kernel = occaBuildKernelFromSource(*device, filename_c, functionName_c, *info);

    OCCA_F2C_FREE_STR(filename    , filename_c);
    OCCA_F2C_FREE_STR(functionName, functionName_c);
  }

  void OCCABUILDKERNELFROMBINARY_FC(occaKernel *kernel, occaDevice *device,
                                    const char *filename     OCCA_F2C_LSTR(filename_l),
                                    const char *functionName OCCA_F2C_LSTR(functionName_l)
                                    OCCA_F2C_RSTR(filename_l)
                                    OCCA_F2C_RSTR(functionName_l)){
    char *filename_c, *functionName_c;
    OCCA_F2C_ALLOC_STR(filename    , filename_l    , filename_c);
    OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

    *kernel = occaBuildKernelFromBinary(*device, filename_c, functionName_c);

    OCCA_F2C_FREE_STR(filename    , filename_c);
    OCCA_F2C_FREE_STR(functionName, functionName_c);
  }

  void OCCABUILDKERNELFROMLOOPY_FC(occaKernel *kernel, occaDevice *device,
                                         const char *filename     OCCA_F2C_LSTR(filename_l),
                                         const char *functionName OCCA_F2C_LSTR(functionName_l),
                                         const char *pythonCode   OCCA_F2C_LSTR(pythonCode_l)
                                         OCCA_F2C_RSTR(filename_l)
                                         OCCA_F2C_RSTR(functionName_l)
                                         OCCA_F2C_RSTR(pythonCode_l)){
    char *filename_c, *functionName_c, *pythonCode_c;

    OCCA_F2C_ALLOC_STR(filename    , filename_l    , filename_c);
    OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);
    OCCA_F2C_ALLOC_STR(pythonCode  , pythonCode_l  , pythonCode_c);

    *kernel = occaBuildKernelFromLoopy(*device, filename_c, functionName_c, pythonCode_c);

    OCCA_F2C_FREE_STR(filename    , filename_c);
    OCCA_F2C_FREE_STR(functionName, functionName_c);
    OCCA_F2C_FREE_STR(pythonCode  , pythonCode_c);
  }

  void OCCADEVICEMALLOC_FC(occaMemory *buf, occaDevice *device,
                           int64_t *bytes, void *source){
    *buf = occaDeviceMalloc(*device, *bytes, source);
  }

  void OCCADEVICEMALLOCNULL_FC(occaMemory *buf, occaDevice *device,
                           int64_t *bytes){
    *buf = occaDeviceMalloc(*device, *bytes, NULL);
  }

  void OCCADEVICEFLUSH_FC(occaDevice *device){
    occaDeviceFlush(*device);
  }
  void OCCADEVICEFINISH_FC(occaDevice *device){
    occaDeviceFinish(*device);
  }

  void OCCADEVICEGENSTREAM_FC(occaStream *stream, occaDevice *device){
    *stream = occaDeviceGenStream(*device);
  }
  void OCCADEVICEGETSTREAM_FC(occaStream *stream, occaDevice *device){
    *stream = occaDeviceGetStream(*device);
  }
  void OCCADEVICESETSTREAM_FC(occaDevice *device, occaStream *stream){
    return occaDeviceSetStream(*device, *stream);
  }

  void OCCADEVICETAGSTREAM_FC(occaTag *tag, occaDevice *device){
    *tag = occaDeviceTagStream(*device);
  }
  void OCCADEVICETIMEBETWEENTAGS_FC(double *time, occaDevice *device,
                                      occaTag *startTag, occaTag *endTag){
    *time = occaDeviceTimeBetweenTags(*device, *startTag, *endTag);
  }

  void OCCADEVICESTREAMFREE_FC(occaDevice *device, occaStream *stream){
    occaDeviceStreamFree(*device, *stream);
  }

  void OCCADEVICEFREE_FC(occaDevice *device){
    occaDeviceFree(*device);
  }
  //====================================


  //---[ Kernel ]-----------------------
  const char* OCCAKERNELMODE_FC(occaKernel* kernel){ //[-]
    return occaKernelMode(kernel);
  }

  void OCCAKERNELPREFERREDDIMSIZE_FC(int32_t *sz, occaKernel *kernel){
    *sz = occaKernelPreferredDimSize(*kernel);
  }

  void OCCAKERNELSETWORKINGDIMS_FC(occaKernel *kernel,
                                   int32_t *dims,
                                   occaDim *items,
                                   occaDim *groups){
    occaKernelSetWorkingDims(*kernel, *dims, *items, *groups);
  }

  void OCCAKERNELSETALLWORKINGDIMS_FC(occaKernel *kernel,
                                      int32_t *dims,
                                      int64_t *itemsX, int64_t *itemsY, int64_t *itemsZ,
                                      int64_t *groupsX, int64_t *groupsY, int64_t *groupsZ){
    occaKernelSetAllWorkingDims(*kernel,
                                *dims,
                                *itemsX, *itemsY, *itemsZ,
                                *groupsX, *groupsY, *groupsZ);
  }

  void OCCAKERNELTIMETAKEN_FC(double *time, occaKernel *kernel){
    *time = occaKernelTimeTaken(kernel);
  }

  void OCCAGENARGUMENTLIST_FC(occaArgumentList *args){
    *args = occaGenArgumentList();
  }

  void OCCAARGUMENTLISTCLEAR_FC(occaArgumentList *list){
    occaArgumentListClear(*list);
  }

  void OCCAARGUMENTLISTFREE_FC(occaArgumentList *list){
    occaArgumentListFree(*list);
  }

  void OCCAARGUMENTLISTADDARGMEM_FC(occaArgumentList *list,
                                    int32_t *argPos,
                                    occaMemory *mem){
    occaArgumentListAddArg(*list, *argPos, mem);
  }

  void OCCAARGUMENTLISTADDARGTYPE_FC(occaArgumentList *list,
                                     int32_t *argPos,
                                     occaType *type){
    occaArgumentListAddArg(*list, *argPos, type);
  }


  void OCCAKERNELRUN__FC(occaKernel *kernel,
                         occaArgumentList *list){
    occaKernelRun_(*kernel, *list);
  }

  void OCCAKERNELFREE_FC(occaKernel *kernel){
    occaKernelFree(*kernel);
  }

  void OCCAGENKERNELINFO_FC(occaKernelInfo *info){
    *info = occaGenKernelInfo();
  }

  void OCCAKERNELINFOADDDEFINE_FC(occaKernelInfo *info,
                                  const char *macro OCCA_F2C_LSTR(macro_l),
                                  occaType *value
                                  OCCA_F2C_RSTR(macro_l)){
    char *macro_c;
    OCCA_F2C_ALLOC_STR(macro, macro_l, macro_c);

    occaKernelInfoAddDefine(*info, macro_c, *value);

    OCCA_F2C_FREE_STR(macro, macro_c);
  }

  void OCCAKERNELINFOFREE_FC(occaKernelInfo *info){
    occaKernelInfoFree(*info);
  }
  //====================================


  //---[ Memory ]-----------------------
  const char* OCCAMEMORYMODE_FC(occaMemory memory){ //[-]
    return occaMemoryMode(memory);
  }

  void OCCACOPYMEMTOMEM_FC(occaMemory *dest, occaMemory *src,
                           const int64_t *bytes,
                           const int64_t *destOffset,
                           const int64_t *srcOffset){
    occaCopyMemToMem(*dest, *src,
                     *bytes, *destOffset, *srcOffset);
  }

  void OCCACOPYPTRTOMEM_FC(occaMemory *dest, const void *src,
                           const int64_t *bytes, const int64_t *offset){
    occaCopyPtrToMem(*dest, src,
                     *bytes, *offset);
  }

  void OCCACOPYMEMTOPTR_FC(void *dest, occaMemory *src,
                           const int64_t *bytes, const int64_t *offset){
    occaCopyMemToPtr(dest, *src,
                     *bytes, *offset);
  }

  void OCCAASYNCCOPYMEMTOMEM_FC(occaMemory *dest, occaMemory *src,
                                const int64_t *bytes,
                                const int64_t *destOffset,
                                const int64_t *srcOffset){
    occaAsyncCopyMemToMem(*dest, *src,
                          *bytes, *destOffset, *srcOffset);
  }

  void OCCAASYNCCOPYPTRTOMEM_FC(occaMemory *dest, const void *src,
                                const int64_t *bytes, const int64_t *offset){
    occaAsyncCopyPtrToMem(*dest, src,
                          *bytes, *offset);
  }

  void OCCAASYNCCOPYMEMTOPTR_FC(void *dest, occaMemory *src,
                                const int64_t *bytes, const int64_t *offset){
    occaAsyncCopyMemToPtr(dest, *src,
                          *bytes, *offset);
  }

  void OCCAMEMORYSWAP_FC(occaMemory *memoryA, occaMemory *memoryB){
    occaMemorySwap(*memoryA, *memoryB);
  }

  void OCCAMEMORYFREE_FC(occaMemory *memory){
    occaMemoryFree(*memory);
  }
  //====================================

#  ifdef __cplusplus
}
#  endif

#endif
