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

#ifndef OCCA_FBASE_HEADER
#define OCCA_FBASE_HEADER

#include <iostream>

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "occa/lang/fortran/defines.hpp"
#include "occa/lang/c/c_wrapper.hpp"
#include "occa/base.hpp"

// [-] Keep [int type] as the first entry
struct occaType_t {
  int type;
  occa::kernelArg_t value;
};

OCCA_START_EXTERN_C

//---[ TypeCasting ]------------------
void OCCASETVERBOSECOMPILATION_FC(const bool value) {
  occaSetVerboseCompilation(value);
}

void OCCAINT32_FC(occaType *type, int32_t *value) {
  if (sizeof(int) == 4) {
    *type = occaInt(*value);
  } else {
    OCCA_ERROR("Bad integer size",
               false);
  }
}

void OCCAINT_FC(occaType *type, int *value) {
  *type = occaInt(*value);
}

void OCCAUINT_FC(occaType *type, unsigned int *value) {
  *type = occaUInt(*value);
}

void OCCACHAR_FC(occaType *type, char *value) {
  *type = occaChar(*value);
}
void OCCAUCHAR_FC(occaType *type, unsigned char *value) {
  *type = occaUChar(*value);
}

void OCCASHORT_FC(occaType *type, short *value) {
  *type = occaShort(*value);
}
void OCCAUSHORT_FC(occaType *type, unsigned short *value) {
  *type = occaUShort(*value);
}

void OCCALONG_FC(occaType *type, long *value) {
  *type = occaLong(*value);
}
void OCCAULONG_FC(occaType *type, unsigned long *value) {
  *type = occaULong(*value);
}

void OCCAFLOAT_FC(occaType *type, float *value) {
  *type = occaFloat(*value);
}
void OCCADOUBLE_FC(occaType *type, double *value) {
  *type = occaDouble(*value);
}

void OCCASTRING_FC(occaType *type, char *str OCCA_F2C_LSTR(str_l)
                   OCCA_F2C_RSTR(str_l)) {
  char *str_c;
  OCCA_F2C_ALLOC_STR(str, str_l, str_c);

  *type = occaString(str_c);

  OCCA_F2C_FREE_STR(str, str_c);
}
//====================================


//---[ Device ]-----------------------
void OCCAPRINTAVAILABLEDEVICES_FC() {
  return occaPrintAvailableDevices();
}

const char* OCCADEVICEMODE_FC(occaDevice device) { // [-]
  return occaDeviceMode(device);
}

void OCCADEVICESETCOMPILER_FC(occaDevice *device,
                              const char *compiler OCCA_F2C_LSTR(compiler_l)
                              OCCA_F2C_RSTR(compiler_l)) {
  char *compiler_c;
  OCCA_F2C_ALLOC_STR(compiler, compiler_l, compiler_c);

  occaDeviceSetCompiler(*device, compiler_c);

  OCCA_F2C_FREE_STR(compiler, compiler_c);
}

void OCCADEVICESETCOMPILERFLAGS_FC(occaDevice *device,
                                   const char *compilerFlags OCCA_F2C_LSTR(compilerFlags_l)
                                   OCCA_F2C_RSTR(compilerFlags_l)) {
  char *compilerFlags_c;
  OCCA_F2C_ALLOC_STR(compilerFlags, compilerFlags_l, compilerFlags_c);

  occaDeviceSetCompilerFlags(*device, compilerFlags_c);

  OCCA_F2C_FREE_STR(compilerFlags, compilerFlags_c);
}

void OCCACREATEDEVICE_FC(occaDevice *device,
                         const char *infos
                         OCCA_F2C_LSTR(infos_l)
                         OCCA_F2C_RSTR(infos_l)) {
  char *infos_c;
  OCCA_F2C_ALLOC_STR(infos, infos_l, infos_c);

  *device = occaCreateDevice(infos_c);

  OCCA_F2C_FREE_STR(infos, infos_c);
}

void OCCACREATEDEVICEFROMINFO_FC(occaDevice *device,
                                 occaDeviceInfo *dInfo) {
  *device = occaCreateDeviceFromInfo(dInfo);
}

void OCCACREATEDEVICEFROMARGS_FC(occaDevice *device, const char *mode OCCA_F2C_LSTR(mode_l),
                                 int32_t *arg1, int32_t *arg2 OCCA_F2C_RSTR(mode_l)) {
  char *mode_c;
  OCCA_F2C_ALLOC_STR(mode, mode_l, mode_c);

  *device = occaCreateDeviceFromArgs(mode_c, *arg1, *arg2);

  OCCA_F2C_FREE_STR(mode, mode_c);
}

void OCCADEVICEMEMORYSIZE_FC(occaDevice *device, int64_t *bytes) {
  *bytes = occaDeviceMemorySize(*device);
}

void OCCADEVICEMEMORYALLOCATED_FC(occaDevice *device, int64_t *bytes) {
  *bytes = occaDeviceMemoryAllocated(*device);
}

// Old version of [OCCADEVICEMEMORYALLOCATED_FC()]
void OCCADEVICEBYTESALLOCATED_FC(occaDevice *device, int64_t *bytes) {
  *bytes = occaDeviceMemoryAllocated(*device);
}

void OCCADEVICEBUILDKERNEL_FC(occaKernel *kernel, occaDevice *device,
                              const char *filename     OCCA_F2C_LSTR(filename_l),
                              const char *functionName OCCA_F2C_LSTR(functionName_l),
                              occaKernelInfo *info
                              OCCA_F2C_RSTR(filename_l)
                              OCCA_F2C_RSTR(functionName_l)) {
  char *filename_c, *functionName_c;
  OCCA_F2C_ALLOC_STR(filename    , filename_l    , filename_c);
  OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

  *kernel = occaDeviceBuildKernel(*device, filename_c, functionName_c, *info);

  OCCA_F2C_FREE_STR(filename    , filename_c);
  OCCA_F2C_FREE_STR(functionName, functionName_c);
}

void OCCADEVICEBUILDKERNELNOKERNELINFO_FC(occaKernel *kernel, occaDevice *device,
                                          const char *filename     OCCA_F2C_LSTR(filename_l),
                                          const char *functionName OCCA_F2C_LSTR(functionName_l)
                                          OCCA_F2C_RSTR(filename_l)
                                          OCCA_F2C_RSTR(functionName_l)) {
  char *filename_c, *functionName_c;
  OCCA_F2C_ALLOC_STR(filename    , filename_l    , filename_c);
  OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

  *kernel = occaDeviceBuildKernel(*device, filename_c, functionName_c, occaNoKernelInfo);

  OCCA_F2C_FREE_STR(filename    , filename_c);
  OCCA_F2C_FREE_STR(functionName, functionName_c);
}

void OCCADEVICEBUILDKERNELFROMSTRING_FC(occaKernel *kernel, occaDevice *device,
                                        const char *str          OCCA_F2C_LSTR(str_l),
                                        const char *functionName OCCA_F2C_LSTR(functionName_l),
                                        occaKernelInfo *info,
                                        const int *language
                                        OCCA_F2C_RSTR(str_l)
                                        OCCA_F2C_RSTR(functionName_l)) {
  char *str_c, *functionName_c;
  OCCA_F2C_ALLOC_STR(str         , str_l         , str_c);
  OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

  *kernel = occaDeviceBuildKernelFromString(*device, str_c, functionName_c, *info, *language);

  OCCA_F2C_FREE_STR(str         , str_c);
  OCCA_F2C_FREE_STR(functionName, functionName_c);
}

void OCCADEVICEBUILDKERNELFROMSTRINGNOARGS_FC(occaKernel *kernel, occaDevice *device,
                                              const char *str          OCCA_F2C_LSTR(str_l),
                                              const char *functionName OCCA_F2C_LSTR(functionName_l)
                                              OCCA_F2C_RSTR(str_l)
                                              OCCA_F2C_RSTR(functionName_l)) {
  char *str_c, *functionName_c;
  OCCA_F2C_ALLOC_STR(str         , str_l         , str_c);
  OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

  *kernel = occaDeviceBuildKernelFromString(*device, str_c, functionName_c, occaNoKernelInfo, occaUsingOKL);

  OCCA_F2C_FREE_STR(str         , str_c);
  OCCA_F2C_FREE_STR(functionName, functionName_c);
}

void OCCADEVICEBUILDKERNELFROMSTRINGNOKERNELINFO_FC(occaKernel *kernel, occaDevice *device,
                                                    const char *str          OCCA_F2C_LSTR(str_l),
                                                    const char *functionName OCCA_F2C_LSTR(functionName_l),
                                                    const int *language
                                                    OCCA_F2C_RSTR(str_l)
                                                    OCCA_F2C_RSTR(functionName_l)) {
  char *str_c, *functionName_c;
  OCCA_F2C_ALLOC_STR(str         , str_l         , str_c);
  OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

  *kernel = occaDeviceBuildKernelFromString(*device, str_c, functionName_c, occaNoKernelInfo, *language);

  OCCA_F2C_FREE_STR(str         , str_c);
  OCCA_F2C_FREE_STR(functionName, functionName_c);
}

void OCCADEVICEBUILDKERNELFROMBINARY_FC(occaKernel *kernel, occaDevice *device,
                                        const char *filename     OCCA_F2C_LSTR(filename_l),
                                        const char *functionName OCCA_F2C_LSTR(functionName_l)
                                        OCCA_F2C_RSTR(filename_l)
                                        OCCA_F2C_RSTR(functionName_l)) {
  char *filename_c, *functionName_c;
  OCCA_F2C_ALLOC_STR(filename    , filename_l    , filename_c);
  OCCA_F2C_ALLOC_STR(functionName, functionName_l, functionName_c);

  *kernel = occaDeviceBuildKernelFromBinary(*device, filename_c, functionName_c);

  OCCA_F2C_FREE_STR(filename    , filename_c);
  OCCA_F2C_FREE_STR(functionName, functionName_c);
}

void OCCADEVICEMALLOC_FC(occaMemory *mem, occaDevice *device,
                         int64_t *bytes, void *source) {
  *mem = occaDeviceMalloc(*device, *bytes, source);
}

void OCCADEVICEMALLOCNULL_FC(occaMemory *mem, occaDevice *device,
                             int64_t *bytes) {
  *mem = occaDeviceMalloc(*device, *bytes, NULL);
}

// void OCCADEVICEMANAGEDALLOC_FC(occaMemory *mem, occaDevice *device,
//                                int64_t *bytes, void *source) {
//   *mem = occaDeviceManagedAlloc(*device, *bytes, source);
// }

// void OCCADEVICEMANAGEDALLOCNULL_FC(occaMemory *mem, occaDevice *device,
//                                    int64_t *bytes) {
//   *mem = occaDeviceManagedAlloc(*device, *bytes, NULL);
// }

void OCCADEVICETEXTUREALLOC_FC(occaMemory *mem,
                               int32_t    *dim,
                               int64_t    *dimX, int64_t *dimY, int64_t *dimZ,
                               void       *source,
                               void       *type, // occaFormatType Missing
                               int32_t    *permissions) {
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
//                                       int32_t    *permissions) {
//   // *mem = occaDeviceManagedTextureAlloc2(*mem,
//   //                                *dim,
//   //                                *dimX, *dimY, *dimZ,
//   //                                source,
//   //                                *type, permissions);
// }

void OCCADEVICEMAPPEDALLOC_FC(occaMemory *mem, occaDevice *device,
                              int64_t *bytes, void *source) {
  *mem = occaDeviceMappedAlloc(*device, *bytes, source);
}

void OCCADEVICEMAPPEDALLOCNULL_FC(occaMemory *mem, occaDevice *device,
                                  int64_t *bytes) {
  *mem = occaDeviceMappedAlloc(*device, *bytes, NULL);
}

// void OCCADEVICEMANAGEDMAPPEDALLOC_FC(occaMemory *mem, occaDevice *device,
//                                      int64_t *bytes, void *source) {
//   *mem = occaDeviceManagedMappedAlloc(*device, *bytes, source);
// }

// void OCCADEVICEMANAGEDMAPPEDALLOCNULL_FC(occaMemory *mem, occaDevice *device,
//                                          int64_t *bytes) {
//   *mem = occaDeviceManagedMappedAlloc(*device, *bytes, NULL);
// }

void OCCADEVICEFLUSH_FC(occaDevice *device) {
  occaDeviceFlush(*device);
}
void OCCADEVICEFINISH_FC(occaDevice *device) {
  occaDeviceFinish(*device);
}

void OCCADEVICECREATESTREAM_FC(occaStream *stream, occaDevice *device) {
  *stream = occaDeviceCreateStream(*device);
}
void OCCADEVICEGETSTREAM_FC(occaStream *stream, occaDevice *device) {
  *stream = occaDeviceGetStream(*device);
}
void OCCADEVICESETSTREAM_FC(occaDevice *device, occaStream *stream) {
  return occaDeviceSetStream(*device, *stream);
}

void OCCADEVICETAGSTREAM_FC(occaStreamTag *tag, occaDevice *device) {
  *tag = occaDeviceTagStream(*device);
}
void OCCADEVICETIMEBETWEENTAGS_FC(double *time, occaDevice *device,
                                  occaStreamTag *startTag, occaStreamTag *endTag) {
  *time = occaDeviceTimeBetweenTags(*device, *startTag, *endTag);
}

void OCCASTREAMFREE_FC(occaStream *stream) {
  occaStreamFree(*stream);
}

void OCCADEVICEFREE_FC(occaDevice *device) {
  occaDeviceFree(*device);
}
//====================================


//---[ Kernel ]-----------------------
const char* OCCAKERNELMODE_FC(occaKernel *kernel) { //[-]
  return occaKernelMode(*kernel);
}

const char* OCCAKERNELNAME_FC(occaKernel *kernel) { //[-]
  return occaKernelName(*kernel);
}

void OCCAKERNELGETDEVICE_FC(occaDevice *device,
                            occaKernel *kernel) {

  *device = occaKernelGetDevice(*kernel);
}

void OCCAKERNELPREFERREDDIMSIZE_FC(int32_t *sz, occaKernel *kernel) {
  *sz = occaKernelPreferredDimSize(*kernel);
}

// void OCCAKERNELSETWORKINGDIMS_FC(occaKernel *kernel,
//                                  int32_t *dims,
//                                  occaDim *items,
//                                  occaDim *groups) {
//   occaKernelSetWorkingDims(*kernel, *dims, *items, *groups);
// }

void OCCAKERNELSETALLWORKINGDIMS_FC(occaKernel *kernel,
                                    int32_t *dims,
                                    int64_t *itemsX, int64_t *itemsY, int64_t *itemsZ,
                                    int64_t *groupsX, int64_t *groupsY, int64_t *groupsZ) {
  occaKernelSetAllWorkingDims(*kernel,
                              *dims,
                              *itemsX, *itemsY, *itemsZ,
                              *groupsX, *groupsY, *groupsZ);
}

void OCCACREATEARGUMENTLIST_FC(occaArgumentList *args) {
  *args = occaCreateArgumentList();
}

void OCCAARGUMENTLISTCLEAR_FC(occaArgumentList *list) {
  occaArgumentListClear(*list);
}

void OCCAARGUMENTLISTFREE_FC(occaArgumentList *list) {
  occaArgumentListFree(*list);
}

void OCCAARGUMENTLISTADDARGMEM_FC(occaArgumentList *list,
                                  int32_t *argPos,
                                  occaMemory *mem) {
  occaArgumentListAddArg(*list, *argPos, *mem);
}

void OCCAARGUMENTLISTADDARGTYPE_FC(occaArgumentList *list,
                                   int32_t *argPos,
                                   occaType *type) {
  occaArgumentListAddArg(*list, *argPos, *type);
}

void OCCAARGUMENTLISTADDARGINT4_FC(occaArgumentList *list,
                                   int32_t *argPos,
                                   int32_t *v) {
  if (sizeof(int) == 4) {
    occaArgumentListAddArg(*list, *argPos, occaInt(*v));
  } else {
    OCCA_ERROR("Bad integer size",
               false);
  }
}

void OCCAARGUMENTLISTADDARGREAL4_FC(occaArgumentList *list,
                                    int32_t *argPos,
                                    float *v) {
  occaArgumentListAddArg(*list, *argPos, occaFloat(*v));
}

void OCCAARGUMENTLISTADDARGREAL8_FC(occaArgumentList *list,
                                    int32_t *argPos,
                                    double *v) {
  occaArgumentListAddArg(*list, *argPos, occaDouble(*v));
}

void OCCAARGUMENTLISTADDARGCHAR_FC(occaArgumentList *list,
                                   int32_t *argPos,
                                   char *v) {
  occaArgumentListAddArg(*list, *argPos, occaChar(*v));
}

void OCCAKERNELRUN01_FC(occaKernel *kernel, occaMemory *arg01) {
  occaKernelRun1(*kernel, *arg01);
}

void OCCAKERNELRUN02_FC(occaKernel *kernel,
                        occaMemory *arg01, occaMemory *arg02) {
  occaKernelRun2(*kernel, *arg01, *arg02);
}

void OCCAKERNELRUN03_FC(occaKernel *kernel,
                        occaMemory *arg01, occaMemory *arg02, occaMemory *arg03) {
  occaKernelRun3(*kernel, *arg01, *arg02, *arg03);
}

void OCCAKERNELRUN04_FC(occaKernel *kernel,
                        occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04) {
  occaKernelRun4(*kernel, *arg01, *arg02, *arg03, *arg04);
}

void OCCAKERNELRUN05_FC(occaKernel *kernel,
                        occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                        occaMemory *arg05) {
  occaKernelRun5(*kernel, *arg01, *arg02, *arg03, *arg04,
                 *arg05);
}

void OCCAKERNELRUN06_FC(occaKernel *kernel,
                        occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                        occaMemory *arg05, occaMemory *arg06) {
  occaKernelRun6(*kernel, *arg01, *arg02, *arg03, *arg04,
                 *arg05, *arg06);
}

void OCCAKERNELRUN07_FC(occaKernel *kernel,
                        occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                        occaMemory *arg05, occaMemory *arg06, occaMemory *arg07) {
  occaKernelRun7(*kernel, *arg01, *arg02, *arg03, *arg04,
                 *arg05, *arg06, *arg07);
}

void OCCAKERNELRUN08_FC(occaKernel *kernel,
                        occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                        occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08) {
  occaKernelRun8(*kernel, *arg01, *arg02, *arg03, *arg04,
                 *arg05, *arg06, *arg07, *arg08);
}

void OCCAKERNELRUN09_FC(occaKernel *kernel,
                        occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                        occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                        occaMemory *arg09) {
  occaKernelRun9(*kernel, *arg01, *arg02, *arg03, *arg04,
                 *arg05, *arg06, *arg07, *arg08,
                 *arg09);
}

void OCCAKERNELRUN10_FC(occaKernel *kernel,
                        occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                        occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                        occaMemory *arg09, occaMemory *arg10) {
  occaKernelRun10(*kernel, *arg01, *arg02, *arg03, *arg04,
                  *arg05, *arg06, *arg07, *arg08,
                  *arg09, *arg10);
}

void OCCAKERNELRUN11_FC(occaKernel *kernel,
                        occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                        occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                        occaMemory *arg09, occaMemory *arg10, occaMemory *arg11) {
  occaKernelRun11(*kernel, *arg01, *arg02, *arg03, *arg04,
                  *arg05, *arg06, *arg07, *arg08,
                  *arg09, *arg10, *arg11);
}

void OCCAKERNELRUN12_FC(occaKernel *kernel,
                        occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                        occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                        occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12) {
  occaKernelRun12(*kernel, *arg01, *arg02, *arg03, *arg04,
                  *arg05, *arg06, *arg07, *arg08,
                  *arg09, *arg10, *arg11, *arg12);
}

void OCCAKERNELRUN13_FC(occaKernel *kernel,
                        occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                        occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                        occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                        occaMemory *arg13) {
  occaKernelRun13(*kernel, *arg01, *arg02, *arg03, *arg04,
                  *arg05, *arg06, *arg07, *arg08,
                  *arg09, *arg10, *arg11, *arg12,
                  *arg13);
}

void OCCAKERNELRUN14_FC(occaKernel *kernel,
                        occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                        occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                        occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                        occaMemory *arg13, occaMemory *arg14) {
  occaKernelRun14(*kernel, *arg01, *arg02, *arg03, *arg04,
                  *arg05, *arg06, *arg07, *arg08,
                  *arg09, *arg10, *arg11, *arg12,
                  *arg13, *arg14);
}

void OCCAKERNELRUN15_FC(occaKernel *kernel,
                        occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                        occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                        occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                        occaMemory *arg13, occaMemory *arg14, occaMemory *arg15) {
  occaKernelRun15(*kernel, *arg01, *arg02, *arg03, *arg04,
                  *arg05, *arg06, *arg07, *arg08,
                  *arg09, *arg10, *arg11, *arg12,
                  *arg13, *arg14, *arg15);
}

void OCCAKERNELRUN16_FC(occaKernel *kernel,
                        occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                        occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                        occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                        occaMemory *arg13, occaMemory *arg14, occaMemory *arg15, occaMemory *arg16) {
  occaKernelRun16(*kernel, *arg01, *arg02, *arg03, *arg04,
                  *arg05, *arg06, *arg07, *arg08,
                  *arg09, *arg10, *arg11, *arg12,
                  *arg13, *arg14, *arg15, *arg16);
}

void OCCAKERNELRUN17_FC(occaKernel *kernel,
                        occaMemory *arg01, occaMemory *arg02, occaMemory *arg03, occaMemory *arg04,
                        occaMemory *arg05, occaMemory *arg06, occaMemory *arg07, occaMemory *arg08,
                        occaMemory *arg09, occaMemory *arg10, occaMemory *arg11, occaMemory *arg12,
                        occaMemory *arg13, occaMemory *arg14, occaMemory *arg15, occaMemory *arg16,
                        occaMemory *arg17) {
  occaKernelRun17(*kernel, *arg01, *arg02, *arg03, *arg04,
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
                        occaMemory *arg17, occaMemory *arg18) {
  occaKernelRun18(*kernel, *arg01, *arg02, *arg03, *arg04,
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
                        occaMemory *arg17, occaMemory *arg18, occaMemory *arg19) {
  occaKernelRun19(*kernel, *arg01, *arg02, *arg03, *arg04,
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
                        occaMemory *arg17, occaMemory *arg18, occaMemory *arg19, occaMemory *arg20) {
  occaKernelRun20(*kernel, *arg01, *arg02, *arg03, *arg04,
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
                        occaMemory *arg21) {
  occaKernelRun21(*kernel, *arg01, *arg02, *arg03, *arg04,
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
                        occaMemory *arg21, occaMemory *arg22) {
  occaKernelRun22(*kernel, *arg01, *arg02, *arg03, *arg04,
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
                        occaMemory *arg21, occaMemory *arg22, occaMemory *arg23) {
  occaKernelRun23(*kernel, *arg01, *arg02, *arg03, *arg04,
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
                        occaMemory *arg21, occaMemory *arg22, occaMemory *arg23, occaMemory *arg24) {
  occaKernelRun24(*kernel, *arg01, *arg02, *arg03, *arg04,
                  *arg05, *arg06, *arg07, *arg08,
                  *arg09, *arg10, *arg11, *arg12,
                  *arg13, *arg14, *arg15, *arg16,
                  *arg17, *arg18, *arg19, *arg20,
                  *arg21, *arg22, *arg23, *arg24);
}

void OCCAKERNELRUN__FC(occaKernel *kernel,
                       occaArgumentList *list) {
  occaKernelRun_(*kernel, *list);
}

void OCCAKERNELFREE_FC(occaKernel *kernel) {
  occaKernelFree(*kernel);
}

void OCCACREATEDEVICEINFO_FC(occaDeviceInfo *info) {
  *info = occaCreateDeviceInfo();
}

void OCCADEVICEINFOAPPEND_FC(occaDeviceInfo *info,
                             const char *key   OCCA_F2C_LSTR(key_l),
                             const char *value OCCA_F2C_LSTR(value_l)
                             OCCA_F2C_RSTR(key_l)
                             OCCA_F2C_RSTR(value_l)) {
  char *key_c, *value_c;
  OCCA_F2C_ALLOC_STR(key, key_l, key_c);
  OCCA_F2C_ALLOC_STR(value, value_l, value_c);

  occaDeviceInfoAppend(*info, key_c, value_c);

  OCCA_F2C_FREE_STR(key, key_c);
  OCCA_F2C_FREE_STR(value, value_c);
}

void OCCADEVICEINFOFREE_FC(occaDeviceInfo *info) {
  occaDeviceInfoFree(*info);
}

void OCCACREATEKERNELINFO_FC(occaKernelInfo *info) {
  *info = occaCreateKernelInfo();
}

void OCCAKERNELINFOADDDEFINE_FC(occaKernelInfo *info,
                                const char *macro OCCA_F2C_LSTR(macro_l),
                                occaType *value
                                OCCA_F2C_RSTR(macro_l)) {
  char *macro_c;
  OCCA_F2C_ALLOC_STR(macro, macro_l, macro_c);

  occaKernelInfoAddDefine(*info, macro_c, *value);

  OCCA_F2C_FREE_STR(macro, macro_c);
}

void OCCAKERNELINFOADDDEFINEINT4_FC(occaKernelInfo *info,
                                    const char *macro OCCA_F2C_LSTR(macro_l),
                                    int32_t *value
                                    OCCA_F2C_RSTR(macro_l)) {
  char *macro_c;
  OCCA_F2C_ALLOC_STR(macro, macro_l, macro_c);

  if (sizeof(int) == 4) {
    occaKernelInfoAddDefine(*info, macro_c, occaInt(*value));
  } else {
    OCCA_ERROR("Bad integer size",
               false);
  }

  OCCA_F2C_FREE_STR(macro, macro_c);
}

void OCCAKERNELINFOADDDEFINEREAL4_FC(occaKernelInfo *info,
                                     const char *macro OCCA_F2C_LSTR(macro_l),
                                     float *value
                                     OCCA_F2C_RSTR(macro_l)) {
  char *macro_c;
  OCCA_F2C_ALLOC_STR(macro, macro_l, macro_c);

  occaKernelInfoAddDefine(*info, macro_c, occaFloat(*value));

  OCCA_F2C_FREE_STR(macro, macro_c);
}

void OCCAKERNELINFOADDDEFINEREAL8_FC(occaKernelInfo *info,
                                     const char *macro OCCA_F2C_LSTR(macro_l),
                                     double *value
                                     OCCA_F2C_RSTR(macro_l)) {
  char *macro_c;
  OCCA_F2C_ALLOC_STR(macro, macro_l, macro_c);

  occaKernelInfoAddDefine(*info, macro_c, occaDouble(*value));

  OCCA_F2C_FREE_STR(macro, macro_c);
}

void OCCAKERNELINFOADDDEFINESTRING_FC(occaKernelInfo *info,
                                      const char *macro OCCA_F2C_LSTR(macro_l),
                                      const char *value OCCA_F2C_LSTR(value_l)
                                      OCCA_F2C_RSTR(macro_l)
                                      OCCA_F2C_RSTR(value_l)
                                      ) {
  char *macro_c, *value_c;
  OCCA_F2C_ALLOC_STR(macro, macro_l, macro_c);
  OCCA_F2C_ALLOC_STR(value, value_l, value_c);

  occaKernelInfoAddDefine(*info, macro_c, occaString(value_c));

  OCCA_F2C_FREE_STR(macro, macro_c);
  OCCA_F2C_FREE_STR(value, value_c);
}


void OCCAKERNELINFOADDINCLUDE_FC(occaKernelInfo *info,
                                 const char *filename OCCA_F2C_LSTR(filename_l)
                                 OCCA_F2C_RSTR(filename_l)) {
  char *filename_c;
  OCCA_F2C_ALLOC_STR(filename, filename_l, filename_c);

  occaKernelInfoAddInclude(*info, filename_c);

  OCCA_F2C_FREE_STR(filename, filename_c);
}

void OCCAKERNELINFOFREE_FC(occaKernelInfo *info) {
  occaKernelInfoFree(*info);
}
//====================================


//---[ Wrappers ]---------------------
void OCCADEVICEWRAPMEMORY_FC(occaMemory *mem, occaDevice *device, void *handle, const int64_t *bytes) {
  *mem = occaDeviceWrapMemory(*device, handle, *bytes);
}

void OCCADEVICEWRAPSTREAM_FC(occaStream *stream, occaDevice *device, void *handle) {
  *stream = occaDeviceWrapStream(*device, handle);
}
//====================================


//---[ Memory ]-----------------------
const char* OCCAMEMORYMODE_FC(occaMemory *memory) { //[-]
  return occaMemoryMode(*memory);
}

const void* OCCAMEMORYGETMAPPEDPOINTER_FC(occaMemory *memory) {
  return occaMemoryGetMappedPointer(*memory);
}

void OCCACOPYMEMTOMEM_FC(occaMemory *dest, occaMemory *src,
                         const int64_t *bytes,
                         const int64_t *destOffset,
                         const int64_t *srcOffset) {
  occaCopyMemToMem(*dest, *src,
                   *bytes, *destOffset, *srcOffset);
}

void OCCACOPYMEMTOMEMAUTO_FC(occaMemory *dest, occaMemory *src) {
  occaCopyMemToMem(*dest, *src, occaAutoSize, occaNoOffset, occaNoOffset);
}

void OCCACOPYPTRTOMEM_FC(occaMemory *dest, const void *src,
                         const int64_t *bytes, const int64_t *offset) {
  occaCopyPtrToMem(*dest, src,
                   *bytes, *offset);
}

void OCCACOPYMEMTOPTR_FC(void *dest, occaMemory *src,
                         const int64_t *bytes, const int64_t *offset) {
  occaCopyMemToPtr(dest, *src,
                   *bytes, *offset);
}

void OCCACOPYPTRTOMEMAUTO_FC(occaMemory *dest, const void *src) {
  occaCopyPtrToMem(*dest, src, occaAutoSize, occaNoOffset);
}

void OCCACOPYMEMTOPTRAUTO_FC(void *dest, occaMemory *src) {
  occaCopyMemToPtr(dest, *src, occaAutoSize, occaNoOffset);
}

void OCCAASYNCCOPYMEMTOMEM_FC(occaMemory *dest, occaMemory *src,
                              const int64_t *bytes,
                              const int64_t *destOffset,
                              const int64_t *srcOffset) {
  occaAsyncCopyMemToMem(*dest, *src,
                        *bytes, *destOffset, *srcOffset);
}

void OCCAASYNCCOPYMEMTOMEMAUTO_FC(occaMemory *dest, occaMemory *src) {
  occaAsyncCopyMemToMem(*dest, *src, occaAutoSize, occaNoOffset, occaNoOffset);
}

void OCCAASYNCCOPYPTRTOMEM_FC(occaMemory *dest, const void *src,
                              const int64_t *bytes, const int64_t *offset) {
  occaAsyncCopyPtrToMem(*dest, src,
                        *bytes, *offset);
}

void OCCAASYNCCOPYMEMTOPTR_FC(void *dest, occaMemory *src,
                              const int64_t *bytes, const int64_t *offset) {
  occaAsyncCopyMemToPtr(dest, *src,
                        *bytes, *offset);
}

void OCCAASYNCCOPYPTRTOMEMAUTO_FC(occaMemory *dest, const void *src) {
  occaAsyncCopyPtrToMem(*dest, src, occaAutoSize, occaNoOffset);
}

void OCCAASYNCCOPYMEMTOPTRAUTO_FC(void *dest, occaMemory *src) {
  occaAsyncCopyMemToPtr(dest, *src, occaAutoSize, occaNoOffset);
}

void OCCAMEMORYSWAP_FC(occaMemory *memoryA, occaMemory *memoryB) {
  occaMemorySwap(*memoryA, *memoryB);
}

void OCCAMEMORYFREE_FC(occaMemory *memory) {
  occaMemoryFree(*memory);
}
//====================================


//---[ Helper Functions ]-------------
void OCCASYSCALL_FC(int32_t *stat,
                    const char *cmdline OCCA_F2C_LSTR(cmdline_l)
                    OCCA_F2C_RSTR(cmdline_l)
                    ) {
  char *cmdline_c;
  OCCA_F2C_ALLOC_STR(cmdline, cmdline_l, cmdline_c);

  *stat = occaSysCall(cmdline_c, NULL);

  OCCA_F2C_FREE_STR(cmdline, cmdline_c);
}
//====================================

OCCA_END_EXTERN_C

#endif
