#ifndef OCCA_FBASE_HEADER
#define OCCA_FBASE_HEADER

#include <iostream>
#include <stdint.h>
#include <stdio.h>

#include "occaDefines.hpp"
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
#define OCCA_F2C_ALLOC_STR(a,n,b)               \
  do {                                          \
    if (a == OCCA_F2C_NULL_CHARACTER_Fortran) { \
      b = 0;                                    \
    } else {                                    \
      while((n > 0) && (a[n-1] == ' ')) n--;    \
      b = (char*)malloc((n+1)*sizeof(char));    \
      if(b==NULL) abort();                      \
      strncpy(b,a,n);                           \
      b[n] = '\0';                              \
    }                                           \
  } while (0)

#define OCCA_F2C_FREE_STR(a,b)                  \
  do {                                          \
    if (a != b) free(b);                        \
  } while (0)

#define OCCA_F2C_GLOBAL_(name,NAME) name##_

#define  OCCASETVERBOSECOMPILATION_FC    OCCA_F2C_GLOBAL_(occasetverbosecompilation_fc  , OCCASETVERBOSECOMPILATION_FC)
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
#define  OCCAPRINTAVAILABLEDEVICES_FC    OCCA_F2C_GLOBAL_(occaprintavailabledevices_fc  , OCCAPRINTAVAILABLEDEVICES_FC)
#define  OCCADEVICEMODE_FC               OCCA_F2C_GLOBAL_(occadevicemode_fc             , OCCADEVICEMODE_FC)
#define  OCCADEVICESETCOMPILER_FC        OCCA_F2C_GLOBAL_(occadevicesetcompiler_fc      , OCCADEVICESETCOMPILER_FC)
#define  OCCADEVICESETCOMPILERFLAGS_FC   OCCA_F2C_GLOBAL_(occadevicesetcompilerflags_fc , OCCADEVICESETCOMPILERFLAGS_FC)
#define  OCCAGETDEVICE_FC                OCCA_F2C_GLOBAL_(occagetdevice_fc              , OCCAGETDEVICE_FC)
#define  OCCAGETDEVICEFROMINFO_FC        OCCA_F2C_GLOBAL_(occagetdevicefrominfo_fc      , OCCAGETDEVICEFROMINFO_FC)
#define  OCCAGETDEVICEFROMARGS_FC        OCCA_F2C_GLOBAL_(occagetdevicefromargs_fc      , OCCAGETDEVICEFROMARGS_FC)
#define  OCCADEVICEBYTESALLOCATED_FC     OCCA_F2C_GLOBAL_(occadevicebytesallocated_fc   , OCCADEVICEBYTESALLOCATED_FC)
#define  OCCABUILDKERNEL_FC              OCCA_F2C_GLOBAL_(occabuildkernel_fc            , OCCABUILDKERNEL_FC)
#define  OCCABUILDKERNELNOKERNELINFO_FC  OCCA_F2C_GLOBAL_(occabuildkernelnokernelinfo_fc, OCCABUILDKERNELNOKERNELINFO_FC)
#define  OCCABUILDKERNELFROMSOURCE_FC    OCCA_F2C_GLOBAL_(occabuildkernelfromsource_fc  , OCCABUILDKERNELFROMSOURCE_FC)
#define  OCCABUILDKERNELFROMSOURCENOKERNELINFO_FC    OCCA_F2C_GLOBAL_(occabuildkernelfromsourcenokernelinfo_fc  , OCCABUILDKERNELFROMSOURCENOKERNELINFO_FC)
#define  OCCABUILDKERNELFROMSTRING_FC    OCCA_F2C_GLOBAL_(occabuildkernelfromstring_fc  , OCCABUILDKERNELFROMSTRING_FC)
#define  OCCABUILDKERNELFROMSTRINGNOKERNELINFO_FC    OCCA_F2C_GLOBAL_(occabuildkernelfromstringnokernelinfo_fc  , OCCABUILDKERNELFROMSTRINGNOKERNELINFO_FC)
#define  OCCABUILDKERNELFROMSTRINGNOARGS_FC    OCCA_F2C_GLOBAL_(occabuildkernelfromstringnoargs_fc  , OCCABUILDKERNELFROMSTRINGNOARGS_FC)
#define  OCCABUILDKERNELFROMBINARY_FC    OCCA_F2C_GLOBAL_(occabuildkernelfrombinary_fc  , OCCABUILDKERNELFROMBINARY_FC)
#define  OCCABUILDKERNELFROMLOOPY_FC     OCCA_F2C_GLOBAL_(occabuildkernelfromloopy_fc   , OCCABUILDKERNELFROMLOOPY_FC)
#define  OCCABUILDKERNELFROMFLOOPY_FC    OCCA_F2C_GLOBAL_(occabuildkernelfromfloopy_fc  , OCCABUILDKERNELFROMFLOOPY_FC)
#define  OCCADEVICEMALLOCNULL_FC         OCCA_F2C_GLOBAL_(occadevicemallocnull_fc       , OCCADEVICEMALLOCNULL_FC)
#define  OCCADEVICEMALLOC_FC             OCCA_F2C_GLOBAL_(occadevicemalloc_fc           , OCCADEVICEMALLOC_FC)
// #define  OCCADEVICEMANAGEDALLOCNULL_FC   OCCA_F2C_GLOBAL_(occadevicemanagedallocnull_fc , OCCADEVICEMANAGEDALLOCNULL_FC)
// #define  OCCADEVICEMANAGEDALLOC_FC       OCCA_F2C_GLOBAL_(occadevicemanagedalloc_fc     , OCCADEVICEMANAGEDALLOC_FC)
// #define  OCCADEVICEUVAALLOCNULL_FC        OCCA_F2C_GLOBAL_(occadeviceuvaallocnull_fc        , OCCADEVICEUVAALLOCNULL_FC)
// #define  OCCADEVICEUVAALLOC_FC            OCCA_F2C_GLOBAL_(occadeviceuvaalloc_fc            , OCCADEVICEUVAALLOC_FC)
// #define  OCCADEVICEMANAGEDUVAALLOCNULL_FC OCCA_F2C_GLOBAL_(occadevicemanageduvaallocnull_fc , OCCADEVICEMANAGEDUVAALLOCNULL_FC)
// #define  OCCADEVICEMANAGEDUVAALLOC_FC      OCCA_F2C_GLOBAL_(occadevicemanageduvaalloc_fc    , OCCADEVICEMANAGEDUVAALLOC_FC)
#define  OCCADEVICETEXTUREALLOC_FC       OCCA_F2C_GLOBAL_(occadevicetexturealloc_fc     , OCCADEVICETEXTUREALLOC_FC)
// #define  OCCADEVICEMANAGEDTEXTUREALLOC_FC OCCA_F2C_GLOBAL_(occadevicemanagedtexturealloc_fc , OCCADEVICEMANAGEDTEXTUREALLOC_FC)
#define  OCCADEVICEMAPPEDALLOCNULL_FC    OCCA_F2C_GLOBAL_(occadevicemappedallocnull_fc  , OCCADEVICEMAPPEDALLOCNULL_FC)
#define  OCCADEVICEMAPPEDALLOC_FC        OCCA_F2C_GLOBAL_(occadevicemappedalloc_fc      , OCCADEVICEMAPPEDALLOC_FC)
// #define  OCCADEVICEMANAGEDMAPPEDALLOCNULL_FC OCCA_F2C_GLOBAL_(occadevicemanagedmappedallocnull_fc , OCCADEVICEMANAGEDMAPPEDALLOCNULL_FC)
// #define  OCCADEVICEMANAGEDMAPPEDALLOC_FC     OCCA_F2C_GLOBAL_(occadevicemanagedmappedalloc_fc     , OCCADEVICEMANAGEDMAPPEDALLOC_FC)
#define  OCCADEVICEFLUSH_FC              OCCA_F2C_GLOBAL_(occadeviceflush_fc            , OCCADEVICEFLUSH_FC)
#define  OCCADEVICEFINISH_FC             OCCA_F2C_GLOBAL_(occadevicefinish_fc           , OCCADEVICEFINISH_FC)
#define  OCCADEVICECREATESTREAM_FC       OCCA_F2C_GLOBAL_(occadevicecreatestream_fc     , OCCADEVICECREATESTREAM_FC)
#define  OCCADEVICEGETSTREAM_FC          OCCA_F2C_GLOBAL_(occadevicegetstream_fc        , OCCADEVICEGETSTREAM_FC)
#define  OCCADEVICESETSTREAM_FC          OCCA_F2C_GLOBAL_(occadevicesetstream_fc        , OCCADEVICESETSTREAM_FC)
#define  OCCADEVICETAGSTREAM_FC          OCCA_F2C_GLOBAL_(occadevicetagstream_fc        , OCCADEVICETAGSTREAM_FC)
#define  OCCADEVICETIMEBETWEENTAGS_FC    OCCA_F2C_GLOBAL_(occadevicetimebetweentags_fc  , OCCADEVICETIMEBETWEENTAGS_FC)
#define  OCCADEVICESTREAMFREE_FC         OCCA_F2C_GLOBAL_(occadevicestreamfree_fc       , OCCADEVICESTREAMFREE_FC)
#define  OCCADEVICEFREE_FC               OCCA_F2C_GLOBAL_(occadevicefree_fc             , OCCADEVICEFREE_FC)
#define  OCCAKERNELMODE_FC               OCCA_F2C_GLOBAL_(occakernelmode_fc             , OCCAKERNELMODE_FC)
#define  OCCAKERNELNAME_FC               OCCA_F2C_GLOBAL_(occakernelname_fc             , OCCAKERNELNAME_FC)
#define  OCCAKERNELGETDEVICE_FC          OCCA_F2C_GLOBAL_(occakernelgetdevice_fc        , OCCAKERNELGETDEVICE_FC)
#define  OCCAKERNELPREFERREDDIMSIZE_FC   OCCA_F2C_GLOBAL_(occakernelpreferreddimsize_fc , OCCAKERNELPREFERREDDIMSIZE_FC)
// #define  OCCAKERNELSETWORKINGDIMS_FC     OCCA_F2C_GLOBAL_(occakernelsetworkingdims_fc   , OCCAKERNELSETWORKINGDIMS_FC)
#define  OCCAKERNELSETALLWORKINGDIMS_FC  OCCA_F2C_GLOBAL_(occakernelsetallworkingdims_fc, OCCAKERNELSETALLWORKINGDIMS_FC)
#define  OCCACREATEARGUMENTLIST_FC       OCCA_F2C_GLOBAL_(occacreateargumentlist_fc     , OCCACREATEARGUMENTLIST_FC)
#define  OCCAARGUMENTLISTCLEAR_FC        OCCA_F2C_GLOBAL_(occaargumentlistclear_fc      , OCCAARGUMENTLISTCLEAR_FC)
#define  OCCAARGUMENTLISTFREE_FC         OCCA_F2C_GLOBAL_(occaargumentlistfree_fc       , OCCAARGUMENTLISTFREE_FC)
#define  OCCAARGUMENTLISTADDARGMEM_FC    OCCA_F2C_GLOBAL_(occaargumentlistaddargmem_fc  , OCCAARGUMENTLISTADDARGMEM_FC)
#define  OCCAARGUMENTLISTADDARGTYPE_FC   OCCA_F2C_GLOBAL_(occaargumentlistaddargtype_fc , OCCAARGUMENTLISTADDARGTYPE_FC)
#define  OCCAARGUMENTLISTADDARGINT4_FC   OCCA_F2C_GLOBAL_(occaargumentlistaddargint4_fc , OCCAARGUMENTLISTADDARGINT4_FC)
#define  OCCAARGUMENTLISTADDARGREAL4_FC  OCCA_F2C_GLOBAL_(occaargumentlistaddargreal4_fc, OCCAARGUMENTLISTADDARGREAL4_FC)
#define  OCCAARGUMENTLISTADDARGREAL8_FC  OCCA_F2C_GLOBAL_(occaargumentlistaddargreal8_fc, OCCAARGUMENTLISTADDARGREAL8_FC)
#define  OCCAARGUMENTLISTADDARGCHAR_FC   OCCA_F2C_GLOBAL_(occaargumentlistaddargchar_fc , OCCAARGUMENTLISTADDARGCHAR_FC)
#define  OCCAKERNELRUN__FC               OCCA_F2C_GLOBAL_(occakernelrun__fc             , OCCAKERNELRUN__FC)
#define  OCCAKERNELRUN01_FC              OCCA_F2C_GLOBAL_(occakernelrun01_fc            , OCCAKERNELRUN01_FC)
#define  OCCAKERNELRUN02_FC              OCCA_F2C_GLOBAL_(occakernelrun02_fc            , OCCAKERNELRUN02_FC)
#define  OCCAKERNELRUN03_FC              OCCA_F2C_GLOBAL_(occakernelrun03_fc            , OCCAKERNELRUN03_FC)
#define  OCCAKERNELRUN04_FC              OCCA_F2C_GLOBAL_(occakernelrun04_fc            , OCCAKERNELRUN04_FC)
#define  OCCAKERNELRUN05_FC              OCCA_F2C_GLOBAL_(occakernelrun05_fc            , OCCAKERNELRUN05_FC)
#define  OCCAKERNELRUN06_FC              OCCA_F2C_GLOBAL_(occakernelrun06_fc            , OCCAKERNELRUN06_FC)
#define  OCCAKERNELRUN07_FC              OCCA_F2C_GLOBAL_(occakernelrun07_fc            , OCCAKERNELRUN07_FC)
#define  OCCAKERNELRUN08_FC              OCCA_F2C_GLOBAL_(occakernelrun08_fc            , OCCAKERNELRUN08_FC)
#define  OCCAKERNELRUN09_FC              OCCA_F2C_GLOBAL_(occakernelrun09_fc            , OCCAKERNELRUN09_FC)
#define  OCCAKERNELRUN10_FC              OCCA_F2C_GLOBAL_(occakernelrun10_fc            , OCCAKERNELRUN10_FC)
#define  OCCAKERNELRUN11_FC              OCCA_F2C_GLOBAL_(occakernelrun11_fc            , OCCAKERNELRUN11_FC)
#define  OCCAKERNELRUN12_FC              OCCA_F2C_GLOBAL_(occakernelrun12_fc            , OCCAKERNELRUN12_FC)
#define  OCCAKERNELRUN13_FC              OCCA_F2C_GLOBAL_(occakernelrun13_fc            , OCCAKERNELRUN13_FC)
#define  OCCAKERNELRUN14_FC              OCCA_F2C_GLOBAL_(occakernelrun14_fc            , OCCAKERNELRUN14_FC)
#define  OCCAKERNELRUN15_FC              OCCA_F2C_GLOBAL_(occakernelrun15_fc            , OCCAKERNELRUN15_FC)
#define  OCCAKERNELRUN16_FC              OCCA_F2C_GLOBAL_(occakernelrun16_fc            , OCCAKERNELRUN16_FC)
#define  OCCAKERNELRUN17_FC              OCCA_F2C_GLOBAL_(occakernelrun17_fc            , OCCAKERNELRUN17_FC)
#define  OCCAKERNELRUN18_FC              OCCA_F2C_GLOBAL_(occakernelrun18_fc            , OCCAKERNELRUN18_FC)
#define  OCCAKERNELRUN19_FC              OCCA_F2C_GLOBAL_(occakernelrun19_fc            , OCCAKERNELRUN19_FC)
#define  OCCAKERNELRUN20_FC              OCCA_F2C_GLOBAL_(occakernelrun20_fc            , OCCAKERNELRUN20_FC)
#define  OCCAKERNELRUN21_FC              OCCA_F2C_GLOBAL_(occakernelrun21_fc            , OCCAKERNELRUN21_FC)
#define  OCCAKERNELRUN22_FC              OCCA_F2C_GLOBAL_(occakernelrun22_fc            , OCCAKERNELRUN22_FC)
#define  OCCAKERNELRUN24_FC              OCCA_F2C_GLOBAL_(occakernelrun24_fc            , OCCAKERNELRUN24_FC)
#define  OCCAKERNELFREE_FC               OCCA_F2C_GLOBAL_(occakernelfree_fc             , OCCAKERNELFREE_FC)
#define  OCCACREATEDEVICEINFO_FC         OCCA_F2C_GLOBAL_(occacreatedeviceinfo_fc       , OCCACREATEDEVICEINFO_FC)
#define  OCCADEVICEINFOAPPEND_FC         OCCA_F2C_GLOBAL_(occadeviceinfoappend_fc       , OCCADEVICEINFOAPPEND_FC)
#define  OCCADEVICEINFOFREE_FC           OCCA_F2C_GLOBAL_(occadeviceinfofree_fc         , OCCADEVICEINFOFREE_FC)
#define  OCCACREATEKERNELINFO_FC         OCCA_F2C_GLOBAL_(occacreatekernelinfo_fc       , OCCACREATEKERNELINFO_FC)
#define  OCCAKERNELINFOADDDEFINE_FC      OCCA_F2C_GLOBAL_(occakernelinfoadddefine_fc    , OCCAKERNELINFOADDDEFINE_FC)
#define  OCCAKERNELINFOADDDEFINEINT4_FC  OCCA_F2C_GLOBAL_(occakernelinfoadddefineint4_fc, OCCAKERNELINFOADDDEFINEINT4_FC)
#define  OCCAKERNELINFOADDDEFINEREAL4_FC OCCA_F2C_GLOBAL_(occakernelinfoadddefinereal4_fc, OCCAKERNELINFOADDDEFINEREAL4_FC)
#define  OCCAKERNELINFOADDDEFINEREAL8_FC OCCA_F2C_GLOBAL_(occakernelinfoadddefinereal8_fc, OCCAKERNELINFOADDDEFINEREAL8_FC)
#define  OCCAKERNELINFOADDDEFINECHAR_FC  OCCA_F2C_GLOBAL_(occakernelinfoadddefinechar_fc, OCCAKERNELINFOADDDEFINECHAR_FC)
#define  OCCAKERNELINFOADDINCLUDE_FC     OCCA_F2C_GLOBAL_(occakernelinfoaddinclude_fc   , OCCAKERNELINFOADDINCLUDE_FC)
#define  OCCAKERNELINFOFREE_FC           OCCA_F2C_GLOBAL_(occakernelinfofree_fc         , OCCAKERNELINFOFREE_FC)
#define  OCCADEVICEWRAPMEMORY_FC         OCCA_F2C_GLOBAL_(occadevicewrapmemory_fc       , OCCADEVICEWRAPMEMORY_FC)
#define  OCCADEVICEWRAPSTREAM_FC         OCCA_F2C_GLOBAL_(occadevicewrapstream_fc       , OCCADEVICEWRAPSTREAM_FC)
#define  OCCAMEMORYMODE_FC               OCCA_F2C_GLOBAL_(occamemorymode_fc             , OCCAMEMORYMODE_FC)
#define  OCCACOPYMEMTOMEM_FC             OCCA_F2C_GLOBAL_(occacopymemtomem_fc           , OCCACOPYMEMTOMEM_FC)
#define  OCCACOPYMEMTOMEMAUTO_FC         OCCA_F2C_GLOBAL_(occacopymemtomemauto_fc       , OCCACOPYMEMTOMEMAUTO_FC)
#define  OCCACOPYPTRTOMEM_FC             OCCA_F2C_GLOBAL_(occacopyptrtomem_fc           , OCCACOPYPTRTOMEM_FC)
#define  OCCACOPYPTRTOMEMAUTO_FC         OCCA_F2C_GLOBAL_(occacopyptrtomemauto_fc       , OCCACOPYPTRTOMEMAUTO_FC)
#define  OCCACOPYMEMTOPTR_FC             OCCA_F2C_GLOBAL_(occacopymemtoptr_fc           , OCCACOPYMEMTOPTR_FC)
#define  OCCACOPYMEMTOPTRAUTO_FC         OCCA_F2C_GLOBAL_(occacopymemtoptrauto_fc       , OCCACOPYMEMTOPTRAUTO_FC)
#define  OCCAASYNCCOPYMEMTOMEM_FC        OCCA_F2C_GLOBAL_(occaasynccopymemtomem_fc      , OCCAASYNCCOPYMEMTOMEM_FC)
#define  OCCAASYNCCOPYMEMTOMEMAUTO_FC    OCCA_F2C_GLOBAL_(occaasynccopymemtomemauto_fc  , OCCAASYNCCOPYMEMTOMEMAUTO_FC)
#define  OCCAASYNCCOPYPTRTOMEM_FC        OCCA_F2C_GLOBAL_(occaasynccopyptrtomem_fc      , OCCAASYNCCOPYPTRTOMEM_FC)
#define  OCCAASYNCCOPYPTRTOMEMAUTO_FC    OCCA_F2C_GLOBAL_(occaasynccopyptrtomemauto_fc  , OCCAASYNCCOPYPTRTOMEMAUTO_FC)
#define  OCCAASYNCCOPYMEMTOPTR_FC        OCCA_F2C_GLOBAL_(occaasynccopymemtoptr_fc      , OCCAASYNCCOPYMEMTOPTR_FC)
#define  OCCAASYNCCOPYMEMTOPTRAUTO_FC    OCCA_F2C_GLOBAL_(occaasynccopymemtoptrauto_fc  , OCCAASYNCCOPYMEMTOPTRAUTO_FC)
#define  OCCAMEMORYSWAP_FC               OCCA_F2C_GLOBAL_(occamemoryswap_fc             , OCCAMEMORYSWAP_FC)
#define  OCCAMEMORYFREE_FC               OCCA_F2C_GLOBAL_(occamemoryfree_fc             , OCCAMEMORYFREE_FC)


#  ifdef __cplusplus
extern "C" {
#  endif


  //---[ TypeCasting ]------------------
  void OCCASETVERBOSECOMPILATION_FC(const bool value){
    occaSetVerboseCompilation(value);
  }

  void OCCAINT32_FC(occaType *type, int32_t *value){
    if(sizeof(int) == 4)
      *type = occaInt(*value);
    else {
      OCCA_CHECK(false, "Bad integer size");
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
  void OCCAPRINTAVAILABLEDEVICES_FC(){
    return occaPrintAvailableDevices();
  }

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

  void OCCAGETDEVICE_FC(occaDevice *device,
                        const char *infos
                        OCCA_F2C_LSTR(infos_l)
                        OCCA_F2C_RSTR(infos_l)){
    char *infos_c;
    OCCA_F2C_ALLOC_STR(infos, infos_l, infos_c);

    *device = occaGetDevice(infos_c);

    OCCA_F2C_FREE_STR(infos, infos_c);
  }

  void OCCAGETDEVICEFROMINFO_FC(occaDevice *device,
                                occaDeviceInfo *dInfo){
    *device = occaGetDeviceFromInfo(dInfo);
  }

  void OCCAGETDEVICEFROMARGS_FC(occaDevice *device, const char *mode OCCA_F2C_LSTR(mode_l),
                                int32_t *arg1, int32_t *arg2 OCCA_F2C_RSTR(mode_l)){
    char *mode_c;
    OCCA_F2C_ALLOC_STR(mode, mode_l, mode_c);

    *device = occaGetDeviceFromArgs(mode_c, *arg1, *arg2);

    OCCA_F2C_FREE_STR(mode, mode_c);
  }

  void OCCADEVICEBYTESALLOCATED_FC(occaDevice *device, int64_t *bytes){
    *bytes = occaDeviceBytesAllocated(*device);
  }

  void OCCABUILDKERNEL_FC(occaKernel *kernel, occaDevice *device,
                          const char *str          OCCA_F2C_LSTR(str_l),
                          const char *functionName OCCA_F2C_LSTR(functionName_l),
                          occaKernelInfo *info
                          OCCA_F2C_RSTR(str_l)
                          OCCA_F2C_RSTR(functionName_l)){
    char *str_c, *functionName_c;
    OCCA_F2C_ALLOC_STR(str         , str_l         , str_c);
    OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

    *kernel = occaBuildKernel(*device, str_c, functionName_c, *info);

    OCCA_F2C_FREE_STR(str         , str_c);
    OCCA_F2C_FREE_STR(functionName, functionName_c);
  }

  void OCCABUILDKERNELNOKERNELINFO_FC(occaKernel *kernel, occaDevice *device,
                                      const char *str          OCCA_F2C_LSTR(str_l),
                                      const char *functionName OCCA_F2C_LSTR(functionName_l)
                                      OCCA_F2C_RSTR(str_l)
                                      OCCA_F2C_RSTR(functionName_l)){
    char *str_c, *functionName_c;
    OCCA_F2C_ALLOC_STR(str         , str_l         , str_c);
    OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

    *kernel = occaBuildKernel(*device, str_c, functionName_c, occaNoKernelInfo);

    OCCA_F2C_FREE_STR(str         , str_c);
    OCCA_F2C_FREE_STR(functionName, functionName_c);
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

  void OCCABUILDKERNELFROMSOURCENOKERNELINFO_FC(occaKernel *kernel, occaDevice *device,
                                                const char *filename     OCCA_F2C_LSTR(filename_l),
                                                const char *functionName OCCA_F2C_LSTR(functionName_l)
                                                OCCA_F2C_RSTR(filename_l)
                                                OCCA_F2C_RSTR(functionName_l)){
    char *filename_c, *functionName_c;
    OCCA_F2C_ALLOC_STR(filename    , filename_l    , filename_c);
    OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

    *kernel = occaBuildKernelFromSource(*device, filename_c, functionName_c, occaNoKernelInfo);

    OCCA_F2C_FREE_STR(filename    , filename_c);
    OCCA_F2C_FREE_STR(functionName, functionName_c);
  }

  void OCCABUILDKERNELFROMSTRING_FC(occaKernel *kernel, occaDevice *device,
                                    const char *str          OCCA_F2C_LSTR(str_l),
                                    const char *functionName OCCA_F2C_LSTR(functionName_l),
                                    occaKernelInfo *info,
                                    const int *language
                                    OCCA_F2C_RSTR(str_l)
                                    OCCA_F2C_RSTR(functionName_l)){
    char *str_c, *functionName_c;
    OCCA_F2C_ALLOC_STR(str         , str_l         , str_c);
    OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

    *kernel = occaBuildKernelFromString(*device, str_c, functionName_c, *info, *language);

    OCCA_F2C_FREE_STR(str         , str_c);
    OCCA_F2C_FREE_STR(functionName, functionName_c);
  }

  void OCCABUILDKERNELFROMSTRINGNOARGS_FC(occaKernel *kernel, occaDevice *device,
                                          const char *str          OCCA_F2C_LSTR(str_l),
                                          const char *functionName OCCA_F2C_LSTR(functionName_l)
                                          OCCA_F2C_RSTR(str_l)
                                          OCCA_F2C_RSTR(functionName_l)){
    char *str_c, *functionName_c;
    OCCA_F2C_ALLOC_STR(str         , str_l         , str_c);
    OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

    *kernel = occaBuildKernelFromString(*device, str_c, functionName_c, occaNoKernelInfo, occaUsingOKL);

    OCCA_F2C_FREE_STR(str         , str_c);
    OCCA_F2C_FREE_STR(functionName, functionName_c);
  }

  void OCCABUILDKERNELFROMSTRINGNOKERNELINFO_FC(occaKernel *kernel, occaDevice *device,
                                                const char *str          OCCA_F2C_LSTR(str_l),
                                                const char *functionName OCCA_F2C_LSTR(functionName_l),
                                                const int *language
                                                OCCA_F2C_RSTR(str_l)
                                                OCCA_F2C_RSTR(functionName_l)){
    char *str_c, *functionName_c;
    OCCA_F2C_ALLOC_STR(str         , str_l         , str_c);
    OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

    *kernel = occaBuildKernelFromString(*device, str_c, functionName_c, occaNoKernelInfo, *language);

    OCCA_F2C_FREE_STR(str         , str_c);
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
                                   occaKernelInfo *info
                                   OCCA_F2C_RSTR(filename_l)
                                   OCCA_F2C_RSTR(functionName_l)
                                   OCCA_F2C_RSTR(pythonCode_l)){
    char *filename_c, *functionName_c;

    OCCA_F2C_ALLOC_STR(filename    , filename_l    , filename_c);
    OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

    *kernel = occaBuildKernelFromLoopy(*device, filename_c, functionName_c, info);

    OCCA_F2C_FREE_STR(filename    , filename_c);
    OCCA_F2C_FREE_STR(functionName, functionName_c);
  }

  void OCCABUILDKERNELFROMFLOOPY_FC(occaKernel *kernel, occaDevice *device,
                                    const char *filename     OCCA_F2C_LSTR(filename_l),
                                    const char *functionName OCCA_F2C_LSTR(functionName_l),
                                    occaKernelInfo *info
                                    OCCA_F2C_RSTR(filename_l)
                                    OCCA_F2C_RSTR(functionName_l)
                                    OCCA_F2C_RSTR(pythonCode_l)){
    char *filename_c, *functionName_c;

    OCCA_F2C_ALLOC_STR(filename    , filename_l    , filename_c);
    OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

    *kernel = occaBuildKernelFromFloopy(*device, filename_c, functionName_c, info);

    OCCA_F2C_FREE_STR(filename    , filename_c);
    OCCA_F2C_FREE_STR(functionName, functionName_c);
  }

  void OCCADEVICEMALLOC_FC(occaMemory *mem, occaDevice *device,
                           int64_t *bytes, void *source){
    *mem = occaDeviceMalloc(*device, *bytes, source);
  }

  void OCCADEVICEMALLOCNULL_FC(occaMemory *mem, occaDevice *device,
                               int64_t *bytes){
    *mem = occaDeviceMalloc(*device, *bytes, NULL);
  }

  // void OCCADEVICEMANAGEDALLOC_FC(occaMemory *mem, occaDevice *device,
  //                                int64_t *bytes, void *source){
  //   *mem = occaDeviceManagedAlloc(*device, *bytes, source);
  // }

  // void OCCADEVICEMANAGEDALLOCNULL_FC(occaMemory *mem, occaDevice *device,
  //                                    int64_t *bytes){
  //   *mem = occaDeviceManagedAlloc(*device, *bytes, NULL);
  // }

  // void OCCADEVICEUVAALLOC_FC(void **ptr, occaDevice *device,
  //                            int64_t *bytes, void *source){
  //   *ptr = occaDeviceUvaAlloc(*device, *bytes, source);
  // }

  // void OCCADEVICEUVAALLOCNULL_FC(void **ptr, occaDevice *device,
  //                                int64_t *bytes){
  //   *ptr = occaDeviceUvaAlloc(*device, *bytes, NULL);
  // }

  // void OCCADEVICEMANAGEDUVAALLOC_FC(void **ptr, occaDevice *device,
  //                                   int64_t *bytes, void *source){
  //   *ptr = occaDeviceManagedUvaAlloc(*device, *bytes, source);
  // }

  // void OCCADEVICEMANAGEDUVAALLOCNULL_FC(void **ptr, occaDevice *device,
  //                                       int64_t *bytes){
  //   *ptr = occaDeviceManagedUvaAlloc(*device, *bytes, NULL);
  // }

  void OCCADEVICETEXTUREALLOC_FC(occaMemory *mem,
                                 int32_t    *dim,
                                 int64_t    *dimX, int64_t *dimY, int64_t *dimZ,
                                 void       *source,
                                 void       *type, // occaFormatType Missing
                                 int32_t    *permissions){
    // *mem = occaDeviceTextureAlloc2(*mem,
    //                                *dim,
    //                                *dimX, *dimY, *dimZ,
    //                                source,
    //                                *type, permissions);
  }

  // void OCCADEVICEMANAGEDTEXTUREALLOC_FC(occaMemory *mem,
  //                                       int32_t    *dim,
  //                                       int64_t    *dimX, int64_t *dimY, int64_t *dimZ,
  //                                       void       *source,
  //                                       void       *type, // occaFormatType Missing
  //                                       int32_t    *permissions){
  //   // *mem = occaDeviceManagedTextureAlloc2(*mem,
  //   //                                *dim,
  //   //                                *dimX, *dimY, *dimZ,
  //   //                                source,
  //   //                                *type, permissions);
  // }

  void OCCADEVICEMAPPEDALLOC_FC(occaMemory *mem, occaDevice *device,
                                int64_t *bytes, void *source){
    *mem = occaDeviceMappedAlloc(*device, *bytes, source);
  }

  void OCCADEVICEMAPPEDALLOCNULL_FC(occaMemory *mem, occaDevice *device,
                                    int64_t *bytes){
    *mem = occaDeviceMappedAlloc(*device, *bytes, NULL);
  }

  // void OCCADEVICEMANAGEDMAPPEDALLOC_FC(occaMemory *mem, occaDevice *device,
  //                                      int64_t *bytes, void *source){
  //   *mem = occaDeviceManagedMappedAlloc(*device, *bytes, source);
  // }

  // void OCCADEVICEMANAGEDMAPPEDALLOCNULL_FC(occaMemory *mem, occaDevice *device,
  //                                          int64_t *bytes){
  //   *mem = occaDeviceManagedMappedAlloc(*device, *bytes, NULL);
  // }

  void OCCADEVICEFLUSH_FC(occaDevice *device){
    occaDeviceFlush(*device);
  }
  void OCCADEVICEFINISH_FC(occaDevice *device){
    occaDeviceFinish(*device);
  }

  void OCCADEVICECREATESTREAM_FC(occaStream *stream, occaDevice *device){
    *stream = occaDeviceCreateStream(*device);
  }
  void OCCADEVICEGETSTREAM_FC(occaStream *stream, occaDevice *device){
    *stream = occaDeviceGetStream(*device);
  }
  void OCCADEVICESETSTREAM_FC(occaDevice *device, occaStream *stream){
    return occaDeviceSetStream(*device, *stream);
  }

  void OCCADEVICETAGSTREAM_FC(occaStreamTag *tag, occaDevice *device){
    *tag = occaDeviceTagStream(*device);
  }
  void OCCADEVICETIMEBETWEENTAGS_FC(double *time, occaDevice *device,
                                    occaStreamTag *startTag, occaStreamTag *endTag){
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
  const char* OCCAKERNELMODE_FC(occaKernel *kernel){ //[-]
    return occaKernelMode(*kernel);
  }

  const char* OCCAKERNELNAME_FC(occaKernel *kernel){ //[-]
    return occaKernelName(*kernel);
  }

  void OCCAKERNELGETDEVICE_FC(occaDevice *device,
                              occaKernel *kernel){

    *device = occaKernelGetDevice(*kernel);
  }

  void OCCAKERNELPREFERREDDIMSIZE_FC(int32_t *sz, occaKernel *kernel){
    *sz = occaKernelPreferredDimSize(*kernel);
  }

  // void OCCAKERNELSETWORKINGDIMS_FC(occaKernel *kernel,
  //                                  int32_t *dims,
  //                                  occaDim *items,
  //                                  occaDim *groups){
  //   occaKernelSetWorkingDims(*kernel, *dims, *items, *groups);
  // }

  void OCCAKERNELSETALLWORKINGDIMS_FC(occaKernel *kernel,
                                      int32_t *dims,
                                      int64_t *itemsX, int64_t *itemsY, int64_t *itemsZ,
                                      int64_t *groupsX, int64_t *groupsY, int64_t *groupsZ){
    occaKernelSetAllWorkingDims(*kernel,
                                *dims,
                                *itemsX, *itemsY, *itemsZ,
                                *groupsX, *groupsY, *groupsZ);
  }

  void OCCACREATEARGUMENTLIST_FC(occaArgumentList *args){
    *args = occaCreateArgumentList();
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
    occaArgumentListAddArg(*list, *argPos, *mem);
  }

  void OCCAARGUMENTLISTADDARGTYPE_FC(occaArgumentList *list,
                                     int32_t *argPos,
                                     occaType *type){
    occaArgumentListAddArg(*list, *argPos, *type);
  }

  void OCCAARGUMENTLISTADDARGINT4_FC(occaArgumentList *list,
                                     int32_t *argPos,
                                     int32_t *v){
    if(sizeof(int) == 4)
    {
      occaArgumentListAddArg(*list, *argPos, occaInt(*v));
    }
    else {
      OCCA_CHECK(false, "Bad integer size");
    }
  }

  void OCCAARGUMENTLISTADDARGREAL4_FC(occaArgumentList *list,
                                     int32_t *argPos,
                                     float *v){
    occaArgumentListAddArg(*list, *argPos, occaFloat(*v));
  }

  void OCCAARGUMENTLISTADDARGREAL8_FC(occaArgumentList *list,
                                     int32_t *argPos,
                                     double *v){
    occaArgumentListAddArg(*list, *argPos, occaDouble(*v));
  }

  void OCCAARGUMENTLISTADDARGCHAR_FC(occaArgumentList *list,
                                     int32_t *argPos,
                                     char *v){
    occaArgumentListAddArg(*list, *argPos, occaChar(*v));
  }

  void OCCAKERNELRUN01_FC(occaKernel *kernel, occaMemory *arg01){
    occaKernelRun(*kernel, *arg01);
  }

  void OCCAKERNELRUN02_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02){
    occaKernelRun(*kernel, *arg01, *arg02);
  }

  void OCCAKERNELRUN03_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03);
  }

  void OCCAKERNELRUN04_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04);
  }

  void OCCAKERNELRUN05_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05);
  }

  void OCCAKERNELRUN06_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06);
  }

  void OCCAKERNELRUN07_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07);
  }

  void OCCAKERNELRUN08_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08);
  }

  void OCCAKERNELRUN09_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                          occaMemory *arg09){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08,
                           *arg09);
  }

  void OCCAKERNELRUN10_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                          occaMemory *arg09, occaMemory *arg10){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08,
                           *arg09, *arg10);
  }

  void OCCAKERNELRUN11_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                          occaMemory *arg09, occaMemory *arg10, occaMemory *arg11){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08,
                           *arg09, *arg10, *arg11);
  }

  void OCCAKERNELRUN12_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                          occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08,
                           *arg09, *arg10, *arg11, *arg12);
  }

  void OCCAKERNELRUN13_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                          occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                          occaMemory *arg13){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08,
                           *arg09, *arg10, *arg11, *arg12,
                           *arg13);
  }

  void OCCAKERNELRUN14_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                          occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                          occaMemory *arg13, occaMemory *arg14){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08,
                           *arg09, *arg10, *arg11, *arg12,
                           *arg13, *arg14);
  }

  void OCCAKERNELRUN15_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                          occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                          occaMemory *arg13, occaMemory *arg14, occaMemory *arg15){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08,
                           *arg09, *arg10, *arg11, *arg12,
                           *arg13, *arg14, *arg15);
  }

  void OCCAKERNELRUN16_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                          occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                          occaMemory *arg13, occaMemory *arg14, occaMemory *arg15, occaMemory *arg16){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08,
                           *arg09, *arg10, *arg11, *arg12,
                           *arg13, *arg14, *arg15, *arg16);
  }

  void OCCAKERNELRUN17_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                          occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                          occaMemory *arg13, occaMemory *arg14, occaMemory *arg15, occaMemory *arg16,
                          occaMemory *arg17){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08,
                           *arg09, *arg10, *arg11, *arg12,
                           *arg13, *arg14, *arg15, *arg16,
                           *arg17);
  }

  void OCCAKERNELRUN18_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                          occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                          occaMemory *arg13, occaMemory *arg14, occaMemory *arg15, occaMemory *arg16,
                          occaMemory *arg17, occaMemory *arg18){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08,
                           *arg09, *arg10, *arg11, *arg12,
                           *arg13, *arg14, *arg15, *arg16,
                           *arg17, *arg18);
  }

  void OCCAKERNELRUN19_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                          occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                          occaMemory *arg13, occaMemory *arg14, occaMemory *arg15, occaMemory *arg16,
                          occaMemory *arg17, occaMemory *arg18, occaMemory *arg19){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08,
                           *arg09, *arg10, *arg11, *arg12,
                           *arg13, *arg14, *arg15, *arg16,
                           *arg17, *arg18, *arg19);
  }

  void OCCAKERNELRUN20_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                          occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                          occaMemory *arg13, occaMemory *arg14, occaMemory *arg15, occaMemory *arg16,
                          occaMemory *arg17, occaMemory *arg18, occaMemory *arg19, occaMemory *arg20){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08,
                           *arg09, *arg10, *arg11, *arg12,
                           *arg13, *arg14, *arg15, *arg16,
                           *arg17, *arg18, *arg19, *arg20);
  }

  void OCCAKERNELRUN21_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                          occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                          occaMemory *arg13, occaMemory *arg14, occaMemory *arg15, occaMemory *arg16,
                          occaMemory *arg17, occaMemory *arg18, occaMemory *arg19, occaMemory *arg20,
                          occaMemory *arg21){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08,
                           *arg09, *arg10, *arg11, *arg12,
                           *arg13, *arg14, *arg15, *arg16,
                           *arg17, *arg18, *arg19, *arg20,
                           *arg21);
  }

  void OCCAKERNELRUN22_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                          occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                          occaMemory *arg13, occaMemory *arg14, occaMemory *arg15, occaMemory *arg16,
                          occaMemory *arg17, occaMemory *arg18, occaMemory *arg19, occaMemory *arg20,
                          occaMemory *arg21, occaMemory *arg22){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08,
                           *arg09, *arg10, *arg11, *arg12,
                           *arg13, *arg14, *arg15, *arg16,
                           *arg17, *arg18, *arg19, *arg20,
                           *arg21, *arg22);
  }

  void OCCAKERNELRUN23_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                          occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                          occaMemory *arg13, occaMemory *arg14, occaMemory *arg15, occaMemory *arg16,
                          occaMemory *arg17, occaMemory *arg18, occaMemory *arg19, occaMemory *arg20,
                          occaMemory *arg21, occaMemory *arg22, occaMemory *arg23){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08,
                           *arg09, *arg10, *arg11, *arg12,
                           *arg13, *arg14, *arg15, *arg16,
                           *arg17, *arg18, *arg19, *arg20,
                           *arg21, *arg22, *arg23);
  }

  void OCCAKERNELRUN24_FC(occaKernel *kernel,
                          occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                          occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                          occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                          occaMemory *arg13, occaMemory *arg14, occaMemory *arg15, occaMemory *arg16,
                          occaMemory *arg17, occaMemory *arg18, occaMemory *arg19, occaMemory *arg20,
                          occaMemory *arg21, occaMemory *arg22, occaMemory *arg23, occaMemory *arg24){
    occaKernelRun(*kernel, *arg01, *arg02, *arg03, *arg04,
                           *arg05, *arg06, *arg07, *arg08,
                           *arg09, *arg10, *arg11, *arg12,
                           *arg13, *arg14, *arg15, *arg16,
                           *arg17, *arg18, *arg19, *arg20,
                           *arg21, *arg22, *arg23, *arg24);
  }

  void OCCAKERNELRUN__FC(occaKernel *kernel,
                         occaArgumentList *list){
    occaKernelRun_(*kernel, *list);
  }

  void OCCAKERNELFREE_FC(occaKernel *kernel){
    occaKernelFree(*kernel);
  }

  void OCCACREATEDEVICEINFO_FC(occaDeviceInfo *info){
    *info = occaCreateDeviceInfo();
  }

  void OCCADEVICEINFOAPPEND_FC(occaDeviceInfo *info,
                               const char *key   OCCA_F2C_LSTR(key_l),
                               const char *value OCCA_F2C_LSTR(value_l)
                               OCCA_F2C_RSTR(key_l)
                               OCCA_F2C_RSTR(value_l)){
    char *key_c, *value_c;
    OCCA_F2C_ALLOC_STR(key, key_l, key_c);
    OCCA_F2C_ALLOC_STR(value, value_l, value_c);

    occaDeviceInfoAppend(*info, key_c, value_c);

    OCCA_F2C_FREE_STR(key, key_c);
    OCCA_F2C_FREE_STR(value, value_c);
  }

  void OCCADEVICEINFOFREE_FC(occaDeviceInfo *info){
    occaDeviceInfoFree(*info);
  }

  void OCCACREATEKERNELINFO_FC(occaKernelInfo *info){
    *info = occaCreateKernelInfo();
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

  void OCCAKERNELINFOADDDEFINEINT4_FC(occaKernelInfo *info,
                                      const char *macro OCCA_F2C_LSTR(macro_l),
                                      int32_t *value
                                      OCCA_F2C_RSTR(macro_l)){
    char *macro_c;
    OCCA_F2C_ALLOC_STR(macro, macro_l, macro_c);

    if(sizeof(int) == 4)
      occaKernelInfoAddDefine(*info, macro_c, occaInt(*value));
    else {
      OCCA_CHECK(false, "Bad integer size");
    }

    OCCA_F2C_FREE_STR(macro, macro_c);
  }

  void OCCAKERNELINFOADDDEFINEREAL4_FC(occaKernelInfo *info,
                                       const char *macro OCCA_F2C_LSTR(macro_l),
                                       float *value
                                       OCCA_F2C_RSTR(macro_l)){
    char *macro_c;
    OCCA_F2C_ALLOC_STR(macro, macro_l, macro_c);

    occaKernelInfoAddDefine(*info, macro_c, occaFloat(*value));

    OCCA_F2C_FREE_STR(macro, macro_c);
  }

  void OCCAKERNELINFOADDDEFINEREAL8_FC(occaKernelInfo *info,
                                       const char *macro OCCA_F2C_LSTR(macro_l),
                                       double *value
                                       OCCA_F2C_RSTR(macro_l)){
    char *macro_c;
    OCCA_F2C_ALLOC_STR(macro, macro_l, macro_c);

    occaKernelInfoAddDefine(*info, macro_c, occaDouble(*value));

    OCCA_F2C_FREE_STR(macro, macro_c);
  }

  void OCCAKERNELINFOADDDEFINECHAR_FC(occaKernelInfo *info,
                                      const char *macro OCCA_F2C_LSTR(macro_l),
                                      char *value
                                      OCCA_F2C_RSTR(macro_l)){
    char *macro_c;
    OCCA_F2C_ALLOC_STR(macro, macro_l, macro_c);

    occaKernelInfoAddDefine(*info, macro_c, occaChar(*value));

    OCCA_F2C_FREE_STR(macro, macro_c);
  }

  void OCCAKERNELINFOADDINCLUDE_FC(occaKernelInfo *info,
                                   const char *filename OCCA_F2C_LSTR(filename_l)
                                   OCCA_F2C_RSTR(filename_l)){
    char *filename_c;
    OCCA_F2C_ALLOC_STR(filename, filename_l, filename_c);

    occaKernelInfoAddInclude(*info, filename_c);

    OCCA_F2C_FREE_STR(filename, filename_c);
  }

  void OCCAKERNELINFOFREE_FC(occaKernelInfo *info){
    occaKernelInfoFree(*info);
  }
  //====================================


  //---[ Wrappers ]---------------------
  void OCCADEVICEWRAPMEMORY_FC(occaMemory *mem, occaDevice *device, void *handle, const int64_t *bytes){
    *mem = occaDeviceWrapMemory(*device, handle, *bytes);
  }

  void OCCADEVICEWRAPSTREAM_FC(occaStream *stream, occaDevice *device, void *handle){
    *stream = occaDeviceWrapStream(*device, handle);
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

  void OCCACOPYMEMTOMEMAUTO_FC(occaMemory *dest, occaMemory *src){
    occaCopyMemToMem(*dest, *src, occaAutoSize, occaNoOffset, occaNoOffset);
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

  void OCCACOPYPTRTOMEMAUTO_FC(occaMemory *dest, const void *src){
    occaCopyPtrToMem(*dest, src, occaAutoSize, occaNoOffset);
  }

  void OCCACOPYMEMTOPTRAUTO_FC(void *dest, occaMemory *src){
    occaCopyMemToPtr(dest, *src, occaAutoSize, occaNoOffset);
  }

  void OCCAASYNCCOPYMEMTOMEM_FC(occaMemory *dest, occaMemory *src,
                                const int64_t *bytes,
                                const int64_t *destOffset,
                                const int64_t *srcOffset){
    occaAsyncCopyMemToMem(*dest, *src,
                          *bytes, *destOffset, *srcOffset);
  }

  void OCCAASYNCCOPYMEMTOMEMAUTO_FC(occaMemory *dest, occaMemory *src){
    occaAsyncCopyMemToMem(*dest, *src, occaAutoSize, occaNoOffset, occaNoOffset);
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

  void OCCAASYNCCOPYPTRTOMEMAUTO_FC(occaMemory *dest, const void *src){
    occaAsyncCopyPtrToMem(*dest, src, occaAutoSize, occaNoOffset);
  }

  void OCCAASYNCCOPYMEMTOPTRAUTO_FC(void *dest, occaMemory *src){
    occaAsyncCopyMemToPtr(dest, *src, occaAutoSize, occaNoOffset);
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
