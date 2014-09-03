#ifndef OCCA_FBASE_HEADER
#define OCCA_FBASE_HEADER

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

#define  OCCAINT_FC                      OCCA_F2C_GLOBAL_(occaInt_fc                    , OCCAINT_FC)
#define  OCCAUINT_FC                     OCCA_F2C_GLOBAL_(occaUInt_fc                   , OCCAUINT_FC)
#define  OCCACHAR_FC                     OCCA_F2C_GLOBAL_(occaChar_fc                   , OCCACHAR_FC)
#define  OCCAUCHAR_FC                    OCCA_F2C_GLOBAL_(occaUChar_fc                  , OCCAUCHAR_FC)
#define  OCCASHORT_FC                    OCCA_F2C_GLOBAL_(occaShort_fc                  , OCCASHORT_FC)
#define  OCCAUSHORT_FC                   OCCA_F2C_GLOBAL_(occaUShort_fc                 , OCCAUSHORT_FC)
#define  OCCALONG_FC                     OCCA_F2C_GLOBAL_(occaLong_fc                   , OCCALONG_FC)
#define  OCCAULONG_FC                    OCCA_F2C_GLOBAL_(occaULong_fc                  , OCCAULONG_FC)
#define  OCCAFLOAT_FC                    OCCA_F2C_GLOBAL_(occaFloat_fc                  , OCCAFLOAT_FC)
#define  OCCADOUBLE_FC                   OCCA_F2C_GLOBAL_(occaDouble_fc                 , OCCADOUBLE_FC)
#define  OCCASTRING_FC                   OCCA_F2C_GLOBAL_(occaString_fc                 , OCCASTRING_FC)
#define  OCCADEVICEMODE_FC               OCCA_F2C_GLOBAL_(occaDeviceMode_fc             , OCCADEVICEMODE_FC)
#define  OCCADEVICESETCOMPILER_FC        OCCA_F2C_GLOBAL_(occaDeviceSetCompiler_fc      , OCCADEVICESETCOMPILER_FC)
#define  OCCADEVICESETCOMPILERFLAGS_FC   OCCA_F2C_GLOBAL_(occaDeviceSetCompilerFlags_fc , OCCADEVICESETCOMPILERFLAGS_FC)
#define  OCCAGETDEVICE_FC                OCCA_F2C_GLOBAL_(occaGetDevice_fc              , OCCAGETDEVICE_FC)
#define  OCCABUILDKERNELFROMSOURCE_FC    OCCA_F2C_GLOBAL_(occaBuildKernelFromSource_fc  , OCCABUILDKERNELFROMSOURCE_FC)
#define  OCCABUILDKERNELFROMBINARY_FC    OCCA_F2C_GLOBAL_(occaBuildKernelFromBinary_fc  , OCCABUILDKERNELFROMBINARY_FC)
#define  OCCABUILDKERNELFROMLOOPY_FC     OCCA_F2C_GLOBAL_(occaBuildKernelFromLoopy_fc   , OCCABUILDKERNELFROMLOOPY_FC)
#define  OCCADEVICEMALLOC_FC             OCCA_F2C_GLOBAL_(occaDeviceMalloc_fc           , OCCADEVICEMALLOC_FC)
#define  OCCADEVICEFLUSH_FC              OCCA_F2C_GLOBAL_(occaDeviceFlush_fc            , OCCADEVICEFLUSH_FC)
#define  OCCADEVICEFINISH_FC             OCCA_F2C_GLOBAL_(occaDeviceFinish_fc           , OCCADEVICEFINISH_FC)
#define  OCCADEVICEGENSTREAM_FC          OCCA_F2C_GLOBAL_(occaDeviceGenStream_fc        , OCCADEVICEGENSTREAM_FC)
#define  OCCADEVICEGETSTREAM_FC          OCCA_F2C_GLOBAL_(occaDeviceGetStream_fc        , OCCADEVICEGETSTREAM_FC)
#define  OCCADEVICESETSTREAM_FC          OCCA_F2C_GLOBAL_(occaDeviceSetStream_fc        , OCCADEVICESETSTREAM_FC)
#define  OCCADEVICETAGSTREAM_FC          OCCA_F2C_GLOBAL_(occaDeviceTagStream_fc        , OCCADEVICETAGSTREAM_FC)
#define  OCCADEVICETIMEBETWEENTAGS_FC    OCCA_F2C_GLOBAL_(occaDeviceTimeBetweenTags_fc  , OCCADEVICETIMEBETWEENTAGS_FC)
#define  OCCADEVICESTREAMFREE_FC         OCCA_F2C_GLOBAL_(occaDeviceStreamFree_fc       , OCCADEVICESTREAMFREE_FC)
#define  OCCADEVICEFREE_FC               OCCA_F2C_GLOBAL_(occaDeviceFree_fc             , OCCADEVICEFREE_FC)
#define  OCCAKERNELMODE_FC               OCCA_F2C_GLOBAL_(occaKernelMode_fc             , OCCAKERNELMODE_FC)
#define  OCCAKERNELPREFERREDDIMSIZE_FC   OCCA_F2C_GLOBAL_(occaKernelPreferredDimSize_fc , OCCAKERNELPREFERREDDIMSIZE_FC)
#define  OCCAKERNELSETWORKINGDIMS_FC     OCCA_F2C_GLOBAL_(occaKernelSetWorkingDims_fc   , OCCAKERNELSETWORKINGDIMS_FC)
#define  OCCAKERNELSETALLWORKINGDIMS_FC  OCCA_F2C_GLOBAL_(occaKernelSetAllWorkingDims_fc, OCCAKERNELSETALLWORKINGDIMS_FC)
#define  OCCAKERNELTIMETAKEN_FC          OCCA_F2C_GLOBAL_(occaKernelTimeTaken_fc        , OCCAKERNELTIMETAKEN_FC)
#define  OCCAGENARGUMENTLIST_FC          OCCA_F2C_GLOBAL_(occaGenArgumentList_fc        , OCCAGENARGUMENTLIST_FC)
#define  OCCAARGUMENTLISTCLEAR_FC        OCCA_F2C_GLOBAL_(occaArgumentListClear_fc      , OCCAARGUMENTLISTCLEAR_FC)
#define  OCCAARGUMENTLISTFREE_FC         OCCA_F2C_GLOBAL_(occaArgumentListFree_fc       , OCCAARGUMENTLISTFREE_FC)
#define  OCCAARGUMENTLISTADDARG_FC       OCCA_F2C_GLOBAL_(occaArgumentListAddArg_fc     , OCCAARGUMENTLISTADDARG_FC)
#define  OCCAKERNELRUN__FC               OCCA_F2C_GLOBAL_(occaKernelRun__fc             , OCCAKERNELRUN__FC)
#define  OCCAKERNELFREE_FC               OCCA_F2C_GLOBAL_(occaKernelFree_fc             , OCCAKERNELFREE_FC)
#define  OCCAGENKERNELINFO_FC            OCCA_F2C_GLOBAL_(occaGenKernelInfo_fc          , OCCAGENKERNELINFO_FC)
#define  OCCAKERNELINFOADDDEFINE_FC      OCCA_F2C_GLOBAL_(occaKernelInfoAddDefine_fc    , OCCAKERNELINFOADDDEFINE_FC)
#define  OCCAKERNELINFOFREE_FC           OCCA_F2C_GLOBAL_(occaKernelInfoFree_fc         , OCCAKERNELINFOFREE_FC)
#define  OCCAMEMORYMODE_FC               OCCA_F2C_GLOBAL_(occaMemoryMode_fc             , OCCAMEMORYMODE_FC)
#define  OCCACOPYMEMTOMEM_FC             OCCA_F2C_GLOBAL_(occaCopyMemToMem_fc           , OCCACOPYMEMTOMEM_FC)
#define  OCCACOPYPTRTOMEM_FC             OCCA_F2C_GLOBAL_(occaCopyPtrToMem_fc           , OCCACOPYPTRTOMEM_FC)
#define  OCCACOPYMEMTOPTR_FC             OCCA_F2C_GLOBAL_(occaCopyMemToPtr_fc           , OCCACOPYMEMTOPTR_FC)
#define  OCCAASYNCCOPYMEMTOMEM_FC        OCCA_F2C_GLOBAL_(occaAsyncCopyMemToMem_fc      , OCCAASYNCCOPYMEMTOMEM_FC)
#define  OCCAASYNCCOPYPTRTOMEM_FC        OCCA_F2C_GLOBAL_(occaAsyncCopyPtrToMem_fc      , OCCAASYNCCOPYPTRTOMEM_FC)
#define  OCCAASYNCCOPYMEMTOPTR_FC        OCCA_F2C_GLOBAL_(occaAsyncCopyMemToPtr_fc      , OCCAASYNCCOPYMEMTOPTR_FC)
#define  OCCAMEMORYSWAP_FC               OCCA_F2C_GLOBAL_(occaMemorySwap_fc             , OCCAMEMORYSWAP_FC)
#define  OCCAMEMORYFREE_FC               OCCA_F2C_GLOBAL_(occaMemoryFree_fc             , OCCAMEMORYFREE_FC)


#  ifdef __cplusplus
extern "C" {
#  endif

  //---[ TypeCasting ]------------------
  occaType OCCAINT_FC(int value){
    return occaInt(value);
  }

  occaType OCCAUINT_FC(unsigned int value){
    return occaUInt(value);
  }

  occaType OCCACHAR_FC(char value){
    return occaChar(value);
  }
  occaType OCCAUCHAR_FC(unsigned char value){
    return occaUChar(value);
  }

  occaType OCCASHORT_FC(short value){
    return occaShort(value);
  }
  occaType OCCAUSHORT_FC(unsigned short value){
    return occaUShort(value);
  }

  occaType OCCALONG_FC(long value){
    return occaLong(value);
  }
  occaType OCCAULONG_FC(unsigned long value){
    return occaULong(value);
  }

  occaType OCCAFLOAT_FC(float value){
    return occaFloat(value);
  }
  occaType OCCADOUBLE_FC(double value){
    return occaDouble(value);
  }

  occaType OCCASTRING_FC(char *str OCCA_F2C_LSTR(str_l)
                         OCCA_F2C_RSTR(str_l)){
    char *str_c;
    OCCA_F2C_ALLOC_STR(str, str_l, str_c);

    occaType ret = occaString(str_c);

    OCCA_F2C_FREE_STR(str, str_c);

    return ret;
  }
  //====================================


  //---[ Device ]-----------------------
  const char* OCCADEVICEMODE_FC(occaDevice device){ // [-]
    occaDeviceMode(device);
  }

  void OCCADEVICESETCOMPILER_FC(occaDevice device,
                                const char *compiler OCCA_F2C_LSTR(compiler_l)
                                OCCA_F2C_RSTR(compiler_l)){
    char *compiler_c;
    OCCA_F2C_ALLOC_STR(compiler, compiler_l, compiler_c);

    occaDeviceSetCompiler(device, compiler_c);

    OCCA_F2C_FREE_STR(compiler, compiler_c);
  }

  void OCCADEVICESETCOMPILERFLAGS_FC(occaDevice device,
                                     const char *compilerFlags OCCA_F2C_LSTR(compilerFlags_l)
                                     OCCA_F2C_RSTR(compilerFlags_l)){
    char *compilerFlags_c;
    OCCA_F2C_ALLOC_STR(compilerFlags, compilerFlags_l, compilerFlags_c);

    occaDeviceSetCompilerFlags(device, compilerFlags_c);

    OCCA_F2C_FREE_STR(compilerFlags, compilerFlags_c);
  }

  occaDevice OCCAGETDEVICE_FC(const char *mode OCCA_F2C_LSTR(mode_l),
                              int arg1, int arg2
                              OCCA_F2C_RSTR(mode_l)){
    char *mode_c;
    OCCA_F2C_ALLOC_STR(mode, mode_l, mode_c);

    occaDevice ret = occaGetDevice(mode_c, arg1, arg2);

    OCCA_F2C_FREE_STR(mode, mode_c);

    return ret;
  }

  occaKernel OCCABUILDKERNELFROMSOURCE_FC(occaDevice device,
                                          const char *filename     OCCA_F2C_LSTR(filename_l),
                                          const char *functionName OCCA_F2C_LSTR(functionName_l),
                                          occaKernelInfo info
                                          OCCA_F2C_RSTR(filename_l)
                                          OCCA_F2C_RSTR(functionName_l)){
    char *filename_c, *functionName_c;
    OCCA_F2C_ALLOC_STR(filename    , filename_l    , filename_c);
    OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

    occaKernel ret = occaBuildKernelFromSource(device, filename_c, functionName_c, info);

    OCCA_F2C_FREE_STR(filename    , filename_c);
    OCCA_F2C_FREE_STR(functionName, functionName_c);

    return ret;
  }

  occaKernel OCCABUILDKERNELFROMBINARY_FC(occaDevice device,
                                          const char *filename     OCCA_F2C_LSTR(filename_l),
                                          const char *functionName OCCA_F2C_LSTR(functionName_l)
                                          OCCA_F2C_RSTR(filename_l)
                                          OCCA_F2C_RSTR(functionName_l)){
    char *filename_c, *functionName_c;
    OCCA_F2C_ALLOC_STR(filename    , filename_l    , filename_c);
    OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

    occaBuildKernelFromBinary(device, filename_c, functionName_c);

    OCCA_F2C_FREE_STR(filename    , filename_c);
    OCCA_F2C_FREE_STR(functionName, functionName_c);
  }

  occaKernel OCCABUILDKERNELFROMLOOPY_FC(occaDevice device,
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

    occaKernel ret = occaBuildKernelFromLoopy(device, filename_c, functionName_c, pythonCode_c);

    OCCA_F2C_FREE_STR(filename    , filename_c);
    OCCA_F2C_FREE_STR(functionName, functionName_c);
    OCCA_F2C_FREE_STR(pythonCode  , pythonCode_c);

    return ret;
  }

  occaMemory OCCADEVICEMALLOC_FC(occaDevice device,
                                 uintptr_t bytes,
                                 void *source){
    return occaDeviceMalloc(device, bytes, source);
  }

  void OCCADEVICEFLUSH_FC(occaDevice device){
    occaDeviceFlush(device);
  }
  void OCCADEVICEFINISH_FC(occaDevice device){
    occaDeviceFinish(device);
  }

  occaStream OCCADEVICEGENSTREAM_FC(occaDevice device){
    return occaDeviceGenStream(device);
  }
  occaStream OCCADEVICEGETSTREAM_FC(occaDevice device){
    return occaDeviceGetStream(device);
  }
  void OCCADEVICESETSTREAM_FC(occaDevice device, occaStream stream){
    return occaDeviceSetStream(device, stream);
  }

  occaTag OCCADEVICETAGSTREAM_FC(occaDevice device){
    return occaDeviceTagStream(device);
  }
  double OCCADEVICETIMEBETWEENTAGS_FC(occaDevice device,
                                      occaTag startTag, occaTag endTag){
    return occaDeviceTimeBetweenTags(device, startTag, endTag);
  }

  void OCCADEVICESTREAMFREE_FC(occaDevice device, occaStream stream){
    occaDeviceStreamFree(device, stream);
  }

  void OCCADEVICEFREE_FC(occaDevice device){
    occaDeviceFree(device);
  }
  //====================================


  //---[ Kernel ]-----------------------
  const char* OCCAKERNELMODE_FC(occaKernel kernel){ //[-]
    occaKernelMode(kernel);
  }

  int OCCAKERNELPREFERREDDIMSIZE_FC(occaKernel kernel){
    return occaKernelPreferredDimSize(kernel);
  }

  void OCCAKERNELSETWORKINGDIMS_FC(occaKernel kernel,
                                   int dims,
                                   occaDim items,
                                   occaDim groups){
    occaKernelSetWorkingDims(kernel, dims, items, groups);
  }

  void OCCAKERNELSETALLWORKINGDIMS_FC(occaKernel kernel,
                                      int dims,
                                      uintptr_t itemsX, uintptr_t itemsY, uintptr_t itemsZ,
                                      uintptr_t groupsX, uintptr_t groupsY, uintptr_t groupsZ){
    occaKernelSetAllWorkingDims(kernel,
                                dims,
                                itemsX, itemsY, itemsZ,
                                groupsX, groupsY, groupsZ);
  }

  double OCCAKERNELTIMETAKEN_FC(occaKernel kernel){
    return occaKernelTimeTaken(kernel);
  }

  occaArgumentList OCCAGENARGUMENTLIST_FC(){
    return occaGenArgumentList();
  }

  void OCCAARGUMENTLISTCLEAR_FC(occaArgumentList list){
    occaArgumentListClear(list);
  }

  void OCCAARGUMENTLISTFREE_FC(occaArgumentList list){
    occaArgumentListFree(list);
  }

  void OCCAARGUMENTLISTADDARG_FC(occaArgumentList list,
                                 int argPos,
                                 void *type){
    occaArgumentListAddArg(list, argPos, type);
  }

  void OCCAKERNELRUN__FC(occaKernel kernel,
                         occaArgumentList list){
    occaKernelRun_(kernel, list);
  }

  void OCCAKERNELFREE_FC(occaKernel kernel){
    occaKernelFree(kernel);
  }

  occaKernelInfo OCCAGENKERNELINFO_FC(){
    return occaGenKernelInfo();
  }

  void OCCAKERNELINFOADDDEFINE_FC(occaKernelInfo info,
                                  const char *macro OCCA_F2C_LSTR(macro_l),
                                  occaType value
                                  OCCA_F2C_RSTR(macro_l)){
    char *macro_c;
    OCCA_F2C_ALLOC_STR(macro, macro_l, macro_c);

    occaKernelInfoAddDefine(info, macro_c, value);

    OCCA_F2C_FREE_STR(macro, macro_c);
  }

  void OCCAKERNELINFOFREE_FC(occaKernelInfo info){
    occaKernelInfoFree(info);
  }
  //====================================


  //---[ Memory ]-----------------------
  const char* OCCAMEMORYMODE_FC(occaMemory memory){ //[-]
    occaMemoryMode(memory);
  }

  void OCCACOPYMEMTOMEM_FC(occaMemory dest, occaMemory src,
                           const uintptr_t bytes,
                           const uintptr_t destOffset,
                           const uintptr_t srcOffset){
    occaCopyMemToMem(dest, src,
                     bytes, destOffset, srcOffset);
  }

  void OCCACOPYPTRTOMEM_FC(occaMemory dest, const void *src,
                           const uintptr_t bytes, const uintptr_t offset){
    occaCopyPtrToMem(dest, src,
                     bytes, offset);
  }

  void OCCACOPYMEMTOPTR_FC(void *dest, occaMemory src,
                           const uintptr_t bytes, const uintptr_t offset){
    occaCopyMemToPtr(dest, src,
                     bytes, offset);
  }

  void OCCAASYNCCOPYMEMTOMEM_FC(occaMemory dest, occaMemory src,
                                const uintptr_t bytes,
                                const uintptr_t destOffset,
                                const uintptr_t srcOffset){
    occaAsyncCopyMemToMem(dest, src,
                          bytes, destOffset, srcOffset);
  }

  void OCCAASYNCCOPYPTRTOMEM_FC(occaMemory dest, const void *src,
                                const uintptr_t bytes, const uintptr_t offset){
    occaAsyncCopyPtrToMem(dest, src,
                          bytes, offset);
  }

  void OCCAASYNCCOPYMEMTOPTR_FC(void *dest, occaMemory src,
                                const uintptr_t bytes, const uintptr_t offset){
    occaAsyncCopyMemToPtr(dest, src,
                          bytes, offset);
  }

  void OCCAMEMORYSWAP_FC(occaMemory memoryA, occaMemory memoryB){
    occaMemorySwap(memoryA, memoryB);
  }

  void OCCAMEMORYFREE_FC(occaMemory memory){
    occaMemoryFree(memory);
  }
  //====================================

#  ifdef __cplusplus
}
#  endif

#endif
