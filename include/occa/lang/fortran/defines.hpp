/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

 */

#define OCCA_F2C_NULL_CHARACTER_Fortran ((char*) 0)

/* --------------------------------------------------------------------*/
/*
    This lets us map the str-len argument either, immediately following
    the char argument (DVF on Win32) or at the end of the argument list
    (general unix compilers)
*/
#if defined(OCCAF_HAVE_FORTRAN_MIXED_STR_ARG)
#  define OCCA_F2C_LSTR(len) , int len
#  define OCCA_F2C_RSTR(len)
#else
#  define OCCA_F2C_LSTR(len)
#  define OCCA_F2C_RSTR(len) , int len
#endif

/* --------------------------------------------------------------------*/
#define OCCA_F2C_ALLOC_STR(a,n,b)               \
  do {                                          \
    if (a == OCCA_F2C_NULL_CHARACTER_Fortran) { \
      b = 0;                                    \
    } else {                                    \
      while((n > 0) && (a[n-1] == ' ')) n--;    \
      b = (char*)malloc((n+1)*sizeof(char));    \
      if (b==NULL) abort();                      \
      strncpy(b,a,n);                           \
      b[n] = '\0';                              \
    }                                           \
  } while (0)

#define OCCA_F2C_FREE_STR(a,b)                  \
  do {                                          \
    if (a != b) free(b);                        \
  } while (0)

#define OCCA_F2C_GLOBAL_(name,NAME) name##_

//---[ Globals & Flags ]----------------
#define  OCCASETVERBOSECOMPILATION_FC                      OCCA_F2C_GLOBAL_(occasetverbosecompilation_fc,\
                                                                            OCCASETVERBOSECOMPILATION_FC)
//======================================

//---[ TypeCasting ]--------------------
#define  OCCAINT_FC                                        OCCA_F2C_GLOBAL_(occaint_fc,\
                                                                            OCCAINT_FC)
#define  OCCAINT32_FC                                      OCCA_F2C_GLOBAL_(occaint32_fc,\
                                                                            OCCAINT32_FC)
#define  OCCAUINT_FC                                       OCCA_F2C_GLOBAL_(occauint_fc,\
                                                                            OCCAUINT_FC)
#define  OCCACHAR_FC                                       OCCA_F2C_GLOBAL_(occachar_fc,\
                                                                            OCCACHAR_FC)
#define  OCCAUCHAR_FC                                      OCCA_F2C_GLOBAL_(occauchar_fc,\
                                                                            OCCAUCHAR_FC)
#define  OCCASHORT_FC                                      OCCA_F2C_GLOBAL_(occashort_fc,\
                                                                            OCCASHORT_FC)
#define  OCCAUSHORT_FC                                     OCCA_F2C_GLOBAL_(occaushort_fc,\
                                                                            OCCAUSHORT_FC)
#define  OCCALONG_FC                                       OCCA_F2C_GLOBAL_(occalong_fc,\
                                                                            OCCALONG_FC)
#define  OCCAULONG_FC                                      OCCA_F2C_GLOBAL_(occaulong_fc,\
                                                                            OCCAULONG_FC)
#define  OCCAFLOAT_FC                                      OCCA_F2C_GLOBAL_(occafloat_fc,\
                                                                            OCCAFLOAT_FC)
#define  OCCADOUBLE_FC                                     OCCA_F2C_GLOBAL_(occadouble_fc,\
                                                                            OCCADOUBLE_FC)
#define  OCCASTRING_FC                                     OCCA_F2C_GLOBAL_(occastring_fc,\
                                                                            OCCASTRING_FC)
//======================================

//---[ Hidden-Device Calls ]------------
//  |---[ Device Functions ]------------
#define OCCASETDEVICE                                      OCCA_F2C_GLOBAL_(occasetdevice_fc, \
                                                                            OCCASETDEVICE_FC)
#define OCCASETDEVICEFROMINFO                              OCCA_F2C_GLOBAL_(occasetdevicefrominfo_fc, \
                                                                            OCCASETDEVICEFROMINFO_FC)
#define OCCAGETCURRENTDEVICE                               OCCA_F2C_GLOBAL_(occagetcurrentdevice_fc, \
                                                                            OCCAGETCURRENTDEVICE_FC)
#define OCCASETCOMPILER                                    OCCA_F2C_GLOBAL_(occasetcompiler_fc, \
                                                                            OCCASETCOMPILER_FC)
#define OCCASETCOMPILERENVSCRIPT                           OCCA_F2C_GLOBAL_(occasetcompilerenvscript_fc, \
                                                                            OCCASETCOMPILERENVSCRIPT_FC)
#define OCCASETCOMPILERFLAGS                               OCCA_F2C_GLOBAL_(occasetcompilerflags_fc, \
                                                                            OCCASETCOMPILERFLAGS_FC)
#define OCCAGETCOMPILER                                    OCCA_F2C_GLOBAL_(occagetcompiler_fc, \
                                                                            OCCAGETCOMPILER_FC)
#define OCCAGETCOMPILERENVSCRIPT                           OCCA_F2C_GLOBAL_(occagetcompilerenvscript_fc, \
                                                                            OCCAGETCOMPILERENVSCRIPT_FC)
#define OCCAGETCOMPILERFLAGS                               OCCA_F2C_GLOBAL_(occagetcompilerflags_fc, \
                                                                            OCCAGETCOMPILERFLAGS_FC)
#define OCCAFLUSH                                          OCCA_F2C_GLOBAL_(occaflush_fc, \
                                                                            OCCAFLUSH_FC)
#define OCCAFINISH                                         OCCA_F2C_GLOBAL_(occafinish_fc,  \
                                                                            OCCAFINISH_FC)
#define OCCAWAITFOR                                        OCCA_F2C_GLOBAL_(occawaitfor_fc, \
                                                                            OCCAWAITFOR_FC)
#define OCCACREATESTREAM                                   OCCA_F2C_GLOBAL_(occacreatestream_fc,  \
                                                                            OCCACREATESTREAM_FC)
#define OCCAGETSTREAM                                      OCCA_F2C_GLOBAL_(occagetstream_fc, \
                                                                            OCCAGETSTREAM_FC)
#define OCCASETSTREAM                                      OCCA_F2C_GLOBAL_(occasetstream_fc, \
                                                                            OCCASETSTREAM_FC)
#define OCCAWRAPSTREAM                                     OCCA_F2C_GLOBAL_(occawrapstream_fc,  \
                                                                            OCCAWRAPSTREAM_FC)
#define OCCATAGSTREAM                                      OCCA_F2C_GLOBAL_(occatagstream_fc, \
                                                                            OCCATAGSTREAM_FC)

//  |---[ Kernel Functions ]------------
#define OCCABUILDKERNEL                                    OCCA_F2C_GLOBAL_(occabuildkernel_fc,  \
                                                                            OCCABUILDKERNEL_FC)
#define OCCABUILDKERNELFROMSTRING                          OCCA_F2C_GLOBAL_(occabuildkernelfromstring_fc, \
                                                                            OCCABUILDKERNELFROMSTRING_FC)
#define OCCABUILDKERNELFROMBINARY                          OCCA_F2C_GLOBAL_(occabuildkernelfrombinary_fc, \
                                                                            OCCABUILDKERNELFROMBINARY_FC)

//  |---[ Memory Functions ]------------
#define OCCAWRAPMEMORY                                     OCCA_F2C_GLOBAL_(occawrapmemory_fc,  \
                                                                            OCCAWRAPMEMORY_FC)
#define OCCAWRAPMANAGEDMEMORY                              OCCA_F2C_GLOBAL_(occawrapmanagedmemory_fc, \
                                                                            OCCAWRAPMANAGEDMEMORY_FC)
#define OCCAMALLOC                                         OCCA_F2C_GLOBAL_(occamalloc_fc,  \
                                                                            OCCAMALLOC_FC)
#define OCCAMANAGEDALLOC                                   OCCA_F2C_GLOBAL_(occamanagedalloc_fc,  \
                                                                            OCCAMANAGEDALLOC_FC)
#define OCCAUVAALLOC                                       OCCA_F2C_GLOBAL_(occauvaalloc_fc,  \
                                                                            OCCAUVAALLOC_FC)
#define OCCAMANAGEDUVAALLOC                                OCCA_F2C_GLOBAL_(occamanageduvaalloc_fc, \
                                                                            OCCAMANAGEDUVAALLOC_FC)
#define OCCAMAPPEDALLOC                                    OCCA_F2C_GLOBAL_(occamappedalloc_fc, \
                                                                            OCCAMAPPEDALLOC_FC)
#define OCCAMANAGEDMAPPEDALLOC                             OCCA_F2C_GLOBAL_(occamanagedmappedalloc_fc, \
                                                                            OCCAMANAGEDMAPPEDALLOC_FC)
//======================================

//---[ Device ]-------------------------
#define  OCCAPRINTAVAILABLEDEVICES_FC                      OCCA_F2C_GLOBAL_(occaprintavailabledevices_fc,\
                                                                            OCCAPRINTAVAILABLEDEVICES_FC)
#define  OCCADEVICEMODE_FC                                 OCCA_F2C_GLOBAL_(occadevicemode_fc,\
                                                                            OCCADEVICEMODE_FC)
#define  OCCADEVICESETCOMPILER_FC                          OCCA_F2C_GLOBAL_(occadevicesetcompiler_fc,\
                                                                            OCCADEVICESETCOMPILER_FC)
#define  OCCADEVICESETCOMPILERFLAGS_FC                     OCCA_F2C_GLOBAL_(occadevicesetcompilerflags_fc,\
                                                                            OCCADEVICESETCOMPILERFLAGS_FC)
#define  OCCACREATEDEVICE_FC                               OCCA_F2C_GLOBAL_(occacreatedevice_fc,\
                                                                            OCCACREATEDEVICE_FC)
#define  OCCACREATEDEVICEFROMINFO_FC                       OCCA_F2C_GLOBAL_(occacreatedevicefrominfo_fc,\
                                                                            OCCACREATEDEVICEFROMINFO_FC)
#define  OCCACREATEDEVICEFROMARGS_FC                       OCCA_F2C_GLOBAL_(occacreatedevicefromargs_fc,\
                                                                            OCCACREATEDEVICEFROMARGS_FC)
#define  OCCADEVICEMEMORYSIZE_FC                           OCCA_F2C_GLOBAL_(occadevicememorysize_fc,\
                                                                            OCCADEVICEMEMORYSIZE_FC)
#define  OCCADEVICEMEMORYALLOCATED_FC                      OCCA_F2C_GLOBAL_(occadevicememoryallocated_fc,\
                                                                            OCCADEVICEMEMORYALLOCATED_FC)
#define  OCCADEVICEBYTESALLOCATED_FC                       OCCA_F2C_GLOBAL_(occadevicebytesallocated_fc,\
                                                                            OCCADEVICEBYTESALLOCATED_FC)
#define  OCCADEVICEBUILDKERNEL_FC                          OCCA_F2C_GLOBAL_(occadevicebuildkernel_fc,\
                                                                            OCCADEVICEBUILDKERNEL_FC)
#define  OCCADEVICEBUILDKERNELNOKERNELINFO_FC              OCCA_F2C_GLOBAL_(occadevicebuildkernelnokernelinfo_fc,\
                                                                            OCCADEVICEBUILDKERNELNOKERNELINFO_FC)
#define  OCCADEVICEBUILDKERNELFROMSTRING_FC                OCCA_F2C_GLOBAL_(occadevicebuildkernelfromstring_fc,\
                                                                            OCCADEVICEBUILDKERNELFROMSTRING_FC)
#define  OCCADEVICEBUILDKERNELFROMSTRINGNOKERNELINFO_FC    OCCA_F2C_GLOBAL_(occadevicebuildkernelfromstringnokernelinfo_fc,\
                                                                            OCCADEVICEBUILDKERNELFROMSTRINGNOKERNELINFO_FC)
#define  OCCADEVICEBUILDKERNELFROMSTRINGNOARGS_FC          OCCA_F2C_GLOBAL_(occadevicebuildkernelfromstringnoargs_fc,\
                                                                            OCCADEVICEBUILDKERNELFROMSTRINGNOARGS_FC)
#define  OCCADEVICEBUILDKERNELFROMBINARY_FC                OCCA_F2C_GLOBAL_(occadevicebuildkernelfrombinary_fc,\
                                                                            OCCADEVICEBUILDKERNELFROMBINARY_FC)
#define  OCCADEVICEMALLOCNULL_FC                           OCCA_F2C_GLOBAL_(occadevicemallocnull_fc,\
                                                                            OCCADEVICEMALLOCNULL_FC)
#define  OCCADEVICEMALLOC_FC                               OCCA_F2C_GLOBAL_(occadevicemalloc_fc,\
                                                                            OCCADEVICEMALLOC_FC)
#define  OCCADEVICEMANAGEDALLOCNULL_FC                     OCCA_F2C_GLOBAL_(occadevicemanagedallocnull_fc, \
                                                                            OCCADEVICEMANAGEDALLOCNULL_FC)
#define  OCCADEVICEMANAGEDALLOC_FC                         OCCA_F2C_GLOBAL_(occadevicemanagedalloc_fc, \
                                                                            OCCADEVICEMANAGEDALLOC_FC)
#define  OCCADEVICEUVAALLOCNULL_FC                         OCCA_F2C_GLOBAL_(occadeviceuvaallocnull_fc, \
                                                                            OCCADEVICEUVAALLOCNULL_FC)
#define  OCCADEVICEUVAALLOC_FC                             OCCA_F2C_GLOBAL_(occadeviceuvaalloc_fc, \
                                                                            OCCADEVICEUVAALLOC_FC)
#define  OCCADEVICEMANAGEDUVAALLOCNULL_FC                  OCCA_F2C_GLOBAL_(occadevicemanageduvaallocnull_fc, \
                                                                            OCCADEVICEMANAGEDUVAALLOCNULL_FC)
#define  OCCADEVICEMANAGEDUVAALLOC_FC                      OCCA_F2C_GLOBAL_(occadevicemanageduvaalloc_fc, \
                                                                            OCCADEVICEMANAGEDUVAALLOC_FC)
#define  OCCADEVICETEXTUREALLOC_FC                         OCCA_F2C_GLOBAL_(occadevicetexturealloc_fc,\
                                                                            OCCADEVICETEXTUREALLOC_FC)
#define  OCCADEVICEMANAGEDTEXTUREALLOC_FC                  OCCA_F2C_GLOBAL_(occadevicemanagedtexturealloc_fc,\
                                                                            OCCADEVICEMANAGEDTEXTUREALLOC_FC)
#define  OCCADEVICEMAPPEDALLOCNULL_FC                      OCCA_F2C_GLOBAL_(occadevicemappedallocnull_fc,\
                                                                            OCCADEVICEMAPPEDALLOCNULL_FC)
#define  OCCADEVICEMAPPEDALLOC_FC                          OCCA_F2C_GLOBAL_(occadevicemappedalloc_fc,\
                                                                            OCCADEVICEMAPPEDALLOC_FC)
#define  OCCADEVICEMANAGEDMAPPEDALLOCNULL_FC               OCCA_F2C_GLOBAL_(occadevicemanagedmappedallocnull_fc,\
                                                                            OCCADEVICEMANAGEDMAPPEDALLOCNULL_FC)
#define  OCCADEVICEMANAGEDMAPPEDALLOC_FC                   OCCA_F2C_GLOBAL_(occadevicemanagedmappedalloc_fc,\
                                                                            OCCADEVICEMANAGEDMAPPEDALLOC_FC)
#define  OCCADEVICEFLUSH_FC                                OCCA_F2C_GLOBAL_(occadeviceflush_fc,\
                                                                            OCCADEVICEFLUSH_FC)
#define  OCCADEVICEFINISH_FC                               OCCA_F2C_GLOBAL_(occadevicefinish_fc,\
                                                                            OCCADEVICEFINISH_FC)
#define  OCCADEVICECREATESTREAM_FC                         OCCA_F2C_GLOBAL_(occadevicecreatestream_fc,\
                                                                            OCCADEVICECREATESTREAM_FC)
#define  OCCADEVICEGETSTREAM_FC                            OCCA_F2C_GLOBAL_(occadevicegetstream_fc,\
                                                                            OCCADEVICEGETSTREAM_FC)
#define  OCCADEVICESETSTREAM_FC                            OCCA_F2C_GLOBAL_(occadevicesetstream_fc,\
                                                                            OCCADEVICESETSTREAM_FC)
#define  OCCADEVICETAGSTREAM_FC                            OCCA_F2C_GLOBAL_(occadevicetagstream_fc,\
                                                                            OCCADEVICETAGSTREAM_FC)
#define  OCCADEVICETIMEBETWEENTAGS_FC                      OCCA_F2C_GLOBAL_(occadevicetimebetweentags_fc,\
                                                                            OCCADEVICETIMEBETWEENTAGS_FC)
#define  OCCADEVICESTREAMFREE_FC                           OCCA_F2C_GLOBAL_(occadevicestreamfree_fc,\
                                                                            OCCADEVICESTREAMFREE_FC)
#define  OCCADEVICEFREE_FC                                 OCCA_F2C_GLOBAL_(occadevicefree_fc,\
                                                                            OCCADEVICEFREE_FC)
//======================================

//---[ Kernel ]-------------------------
#define  OCCAKERNELMODE_FC                                 OCCA_F2C_GLOBAL_(occakernelmode_fc,\
                                                                            OCCAKERNELMODE_FC)
#define  OCCAKERNELNAME_FC                                 OCCA_F2C_GLOBAL_(occakernelname_fc,\
                                                                            OCCAKERNELNAME_FC)
#define  OCCAKERNELGETDEVICE_FC                            OCCA_F2C_GLOBAL_(occakernelgetdevice_fc,\
                                                                            OCCAKERNELGETDEVICE_FC)
#define  OCCAKERNELPREFERREDDIMSIZE_FC                     OCCA_F2C_GLOBAL_(occakernelpreferreddimsize_fc,\
                                                                            OCCAKERNELPREFERREDDIMSIZE_FC)
#define  OCCAKERNELSETALLWORKINGDIMS_FC                    OCCA_F2C_GLOBAL_(occakernelsetallworkingdims_fc,\
                                                                            OCCAKERNELSETALLWORKINGDIMS_FC)
#define  OCCACREATEARGUMENTLIST_FC                         OCCA_F2C_GLOBAL_(occacreateargumentlist_fc,\
                                                                            OCCACREATEARGUMENTLIST_FC)
#define  OCCAARGUMENTLISTCLEAR_FC                          OCCA_F2C_GLOBAL_(occaargumentlistclear_fc,\
                                                                            OCCAARGUMENTLISTCLEAR_FC)
#define  OCCAARGUMENTLISTFREE_FC                           OCCA_F2C_GLOBAL_(occaargumentlistfree_fc,\
                                                                            OCCAARGUMENTLISTFREE_FC)
#define  OCCAARGUMENTLISTADDARGMEM_FC                      OCCA_F2C_GLOBAL_(occaargumentlistaddargmem_fc,\
                                                                            OCCAARGUMENTLISTADDARGMEM_FC)
#define  OCCAARGUMENTLISTADDARGTYPE_FC                     OCCA_F2C_GLOBAL_(occaargumentlistaddargtype_fc,\
                                                                            OCCAARGUMENTLISTADDARGTYPE_FC)
#define  OCCAARGUMENTLISTADDARGINT4_FC                     OCCA_F2C_GLOBAL_(occaargumentlistaddargint4_fc,\
                                                                            OCCAARGUMENTLISTADDARGINT4_FC)
#define  OCCAARGUMENTLISTADDARGREAL4_FC                    OCCA_F2C_GLOBAL_(occaargumentlistaddargreal4_fc,\
                                                                            OCCAARGUMENTLISTADDARGREAL4_FC)
#define  OCCAARGUMENTLISTADDARGREAL8_FC                    OCCA_F2C_GLOBAL_(occaargumentlistaddargreal8_fc,\
                                                                            OCCAARGUMENTLISTADDARGREAL8_FC)
#define  OCCAARGUMENTLISTADDARGCHAR_FC                     OCCA_F2C_GLOBAL_(occaargumentlistaddargchar_fc,\
                                                                            OCCAARGUMENTLISTADDARGCHAR_FC)
#define  OCCAKERNELRUN__FC                                 OCCA_F2C_GLOBAL_(occakernelrun__fc,\
                                                                            OCCAKERNELRUN__FC)
#define  OCCAKERNELRUN01_FC                                OCCA_F2C_GLOBAL_(occakernelrun01_fc,\
                                                                            OCCAKERNELRUN01_FC)
#define  OCCAKERNELRUN02_FC                                OCCA_F2C_GLOBAL_(occakernelrun02_fc,\
                                                                            OCCAKERNELRUN02_FC)
#define  OCCAKERNELRUN03_FC                                OCCA_F2C_GLOBAL_(occakernelrun03_fc,\
                                                                            OCCAKERNELRUN03_FC)
#define  OCCAKERNELRUN04_FC                                OCCA_F2C_GLOBAL_(occakernelrun04_fc,\
                                                                            OCCAKERNELRUN04_FC)
#define  OCCAKERNELRUN05_FC                                OCCA_F2C_GLOBAL_(occakernelrun05_fc,\
                                                                            OCCAKERNELRUN05_FC)
#define  OCCAKERNELRUN06_FC                                OCCA_F2C_GLOBAL_(occakernelrun06_fc,\
                                                                            OCCAKERNELRUN06_FC)
#define  OCCAKERNELRUN07_FC                                OCCA_F2C_GLOBAL_(occakernelrun07_fc,\
                                                                            OCCAKERNELRUN07_FC)
#define  OCCAKERNELRUN08_FC                                OCCA_F2C_GLOBAL_(occakernelrun08_fc,\
                                                                            OCCAKERNELRUN08_FC)
#define  OCCAKERNELRUN09_FC                                OCCA_F2C_GLOBAL_(occakernelrun09_fc,\
                                                                            OCCAKERNELRUN09_FC)
#define  OCCAKERNELRUN10_FC                                OCCA_F2C_GLOBAL_(occakernelrun10_fc,\
                                                                            OCCAKERNELRUN10_FC)
#define  OCCAKERNELRUN11_FC                                OCCA_F2C_GLOBAL_(occakernelrun11_fc,\
                                                                            OCCAKERNELRUN11_FC)
#define  OCCAKERNELRUN12_FC                                OCCA_F2C_GLOBAL_(occakernelrun12_fc,\
                                                                            OCCAKERNELRUN12_FC)
#define  OCCAKERNELRUN13_FC                                OCCA_F2C_GLOBAL_(occakernelrun13_fc,\
                                                                            OCCAKERNELRUN13_FC)
#define  OCCAKERNELRUN14_FC                                OCCA_F2C_GLOBAL_(occakernelrun14_fc,\
                                                                            OCCAKERNELRUN14_FC)
#define  OCCAKERNELRUN15_FC                                OCCA_F2C_GLOBAL_(occakernelrun15_fc,\
                                                                            OCCAKERNELRUN15_FC)
#define  OCCAKERNELRUN16_FC                                OCCA_F2C_GLOBAL_(occakernelrun16_fc,\
                                                                            OCCAKERNELRUN16_FC)
#define  OCCAKERNELRUN17_FC                                OCCA_F2C_GLOBAL_(occakernelrun17_fc,\
                                                                            OCCAKERNELRUN17_FC)
#define  OCCAKERNELRUN18_FC                                OCCA_F2C_GLOBAL_(occakernelrun18_fc,\
                                                                            OCCAKERNELRUN18_FC)
#define  OCCAKERNELRUN19_FC                                OCCA_F2C_GLOBAL_(occakernelrun19_fc,\
                                                                            OCCAKERNELRUN19_FC)
#define  OCCAKERNELRUN20_FC                                OCCA_F2C_GLOBAL_(occakernelrun20_fc,\
                                                                            OCCAKERNELRUN20_FC)
#define  OCCAKERNELRUN21_FC                                OCCA_F2C_GLOBAL_(occakernelrun21_fc,\
                                                                            OCCAKERNELRUN21_FC)
#define  OCCAKERNELRUN22_FC                                OCCA_F2C_GLOBAL_(occakernelrun22_fc,\
                                                                            OCCAKERNELRUN22_FC)
#define  OCCAKERNELRUN24_FC                                OCCA_F2C_GLOBAL_(occakernelrun24_fc,\
                                                                            OCCAKERNELRUN24_FC)
#define  OCCAKERNELFREE_FC                                 OCCA_F2C_GLOBAL_(occakernelfree_fc,\
                                                                            OCCAKERNELFREE_FC)
#define  OCCACREATEDEVICEINFO_FC                           OCCA_F2C_GLOBAL_(occacreatedeviceinfo_fc,\
                                                                            OCCACREATEDEVICEINFO_FC)
#define  OCCADEVICEINFOAPPEND_FC                           OCCA_F2C_GLOBAL_(occadeviceinfoappend_fc,\
                                                                            OCCADEVICEINFOAPPEND_FC)
#define  OCCADEVICEINFOFREE_FC                             OCCA_F2C_GLOBAL_(occadeviceinfofree_fc,\
                                                                            OCCADEVICEINFOFREE_FC)
#define  OCCACREATEKERNELINFO_FC                           OCCA_F2C_GLOBAL_(occacreatekernelinfo_fc,\
                                                                            OCCACREATEKERNELINFO_FC)
#define  OCCAKERNELINFOADDDEFINE_FC                        OCCA_F2C_GLOBAL_(occakernelinfoadddefine_fc,\
                                                                            OCCAKERNELINFOADDDEFINE_FC)
#define  OCCAKERNELINFOADDDEFINEINT4_FC                    OCCA_F2C_GLOBAL_(occakernelinfoadddefineint4_fc,\
                                                                            OCCAKERNELINFOADDDEFINEINT4_FC)
#define  OCCAKERNELINFOADDDEFINEREAL4_FC                   OCCA_F2C_GLOBAL_(occakernelinfoadddefinereal4_fc,\
                                                                            OCCAKERNELINFOADDDEFINEREAL4_FC)
#define  OCCAKERNELINFOADDDEFINEREAL8_FC                   OCCA_F2C_GLOBAL_(occakernelinfoadddefinereal8_fc,\
                                                                            OCCAKERNELINFOADDDEFINEREAL8_FC)
#define  OCCAKERNELINFOADDDEFINESTRING_FC                  OCCA_F2C_GLOBAL_(occakernelinfoadddefinestring_fc,\
                                                                            OCCAKERNELINFOADDDEFINESTRING_FC)
#define  OCCAKERNELINFOADDINCLUDE_FC                       OCCA_F2C_GLOBAL_(occakernelinfoaddinclude_fc,\
                                                                            OCCAKERNELINFOADDINCLUDE_FC)
#define  OCCAKERNELINFOFREE_FC                             OCCA_F2C_GLOBAL_(occakernelinfofree_fc,\
                                                                            OCCAKERNELINFOFREE_FC)
#define  OCCADEVICEWRAPMEMORY_FC                           OCCA_F2C_GLOBAL_(occadevicewrapmemory_fc,\
                                                                            OCCADEVICEWRAPMEMORY_FC)
#define  OCCADEVICEWRAPSTREAM_FC                           OCCA_F2C_GLOBAL_(occadevicewrapstream_fc,\
                                                                            OCCADEVICEWRAPSTREAM_FC)
//======================================

//---[ Memory ]-------------------------
#define  OCCAMEMORYMODE_FC                                 OCCA_F2C_GLOBAL_(occamemorymode_fc,\
                                                                            OCCAMEMORYMODE_FC)
#define  OCCAMEMORYSWAP_FC                                 OCCA_F2C_GLOBAL_(occamemoryswap_fc,\
                                                                            OCCAMEMORYSWAP_FC)
#define  OCCAMEMORYGETMAPPEDPOINTER_FC                     OCCA_F2C_GLOBAL_(occamemorygetmappedpointer_fc,\
                                                                            OCCAMEMORYGETMAPPEDPOINTER_FC)
#define  OCCACOPYMEMTOMEM_FC                               OCCA_F2C_GLOBAL_(occacopymemtomem_fc,\
                                                                            OCCACOPYMEMTOMEM_FC)
#define  OCCACOPYMEMTOMEMAUTO_FC                           OCCA_F2C_GLOBAL_(occacopymemtomemauto_fc,\
                                                                            OCCACOPYMEMTOMEMAUTO_FC)
#define  OCCACOPYPTRTOMEM_FC                               OCCA_F2C_GLOBAL_(occacopyptrtomem_fc,\
                                                                            OCCACOPYPTRTOMEM_FC)
#define  OCCACOPYPTRTOMEMAUTO_FC                           OCCA_F2C_GLOBAL_(occacopyptrtomemauto_fc,\
                                                                            OCCACOPYPTRTOMEMAUTO_FC)
#define  OCCACOPYMEMTOPTR_FC                               OCCA_F2C_GLOBAL_(occacopymemtoptr_fc,\
                                                                            OCCACOPYMEMTOPTR_FC)
#define  OCCACOPYMEMTOPTRAUTO_FC                           OCCA_F2C_GLOBAL_(occacopymemtoptrauto_fc,\
                                                                            OCCACOPYMEMTOPTRAUTO_FC)
#define  OCCAASYNCCOPYMEMTOMEM_FC                          OCCA_F2C_GLOBAL_(occaasynccopymemtomem_fc,\
                                                                            OCCAASYNCCOPYMEMTOMEM_FC)
#define  OCCAASYNCCOPYMEMTOMEMAUTO_FC                      OCCA_F2C_GLOBAL_(occaasynccopymemtomemauto_fc,\
                                                                            OCCAASYNCCOPYMEMTOMEMAUTO_FC)
#define  OCCAASYNCCOPYPTRTOMEM_FC                          OCCA_F2C_GLOBAL_(occaasynccopyptrtomem_fc,\
                                                                            OCCAASYNCCOPYPTRTOMEM_FC)
#define  OCCAASYNCCOPYPTRTOMEMAUTO_FC                      OCCA_F2C_GLOBAL_(occaasynccopyptrtomemauto_fc,\
                                                                            OCCAASYNCCOPYPTRTOMEMAUTO_FC)
#define  OCCAASYNCCOPYMEMTOPTR_FC                          OCCA_F2C_GLOBAL_(occaasynccopymemtoptr_fc,\
                                                                            OCCAASYNCCOPYMEMTOPTR_FC)
#define  OCCAASYNCCOPYMEMTOPTRAUTO_FC                      OCCA_F2C_GLOBAL_(occaasynccopymemtoptrauto_fc,\
                                                                            OCCAASYNCCOPYMEMTOPTRAUTO_FC)
#define  OCCAMEMORYFREE_FC                                 OCCA_F2C_GLOBAL_(occamemoryfree_fc,\
                                                                            OCCAMEMORYFREE_FC)
//======================================

//---[ Helper Functions ]---------------
#define  OCCASYSCALL_FC                                    OCCA_F2C_GLOBAL_(occasyscall_fc,\
                                                                            OCCASYSCALL_FC)
//======================================
