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

#ifndef OCCA_CBASE_HEADER
#define OCCA_CBASE_HEADER

#include "stdint.h"
#include "stdlib.h"

#include "occa/defines.hpp"

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
#  define OCCA_RFUNC
#  define OCCA_LFUNC
#else
#  define OCCA_RFUNC __stdcall
#  ifdef OCCA_C_EXPORTS
//   #define OCCA_LFUNC __declspec(dllexport)
#    define OCCA_LFUNC
#  else
//   #define OCCA_LFUNC __declspec(dllimport)
#    define OCCA_LFUNC
#  endif
#endif

OCCA_START_EXTERN_C

struct occaObject_t;
struct occaObject {
  struct occaObject_t *ptr;
};

typedef struct occaObject occaType;
typedef struct occaObject occaDevice;
typedef struct occaObject occaKernel;
typedef struct occaObject occaMemory;
typedef struct occaObject occaStream;
typedef struct occaObject occaProperties;
typedef struct occaObject occaArgumentList;

typedef struct occaStreamTag_t {
  double tagTime;
  void *handle;
} occaStreamTag;

typedef int64_t  occaDim_t;
typedef uint64_t occaUDim_t;

typedef struct {
  occaUDim_t x, y, z;
} occaDim;

typedef enum {
  IR  = OCCA_CONST_IR,
  OKL = OCCA_CONST_OKL,
  OFL = OCCA_CONST_OFL
} occaLanguage;


//---[ Globals & Flags ]----------------
extern OCCA_LFUNC const occaObject occaDefault;
extern OCCA_LFUNC const occaUDim_t occaAllBytes;
extern OCCA_LFUNC const void *occaEmptyProperties;
//======================================


//---[ Types ]--------------------------
//  ---[ Known Types ]------------------
OCCA_LFUNC occaType OCCA_RFUNC occaPtr(void *value);

OCCA_LFUNC occaType OCCA_RFUNC occaInt8(int8_t value);
OCCA_LFUNC occaType OCCA_RFUNC occaUInt8(uint8_t value);

OCCA_LFUNC occaType OCCA_RFUNC occaInt16(int16_t value);
OCCA_LFUNC occaType OCCA_RFUNC occaUInt16(uint16_t value);

OCCA_LFUNC occaType OCCA_RFUNC occaInt32(int32_t value);
OCCA_LFUNC occaType OCCA_RFUNC occaUInt32(uint32_t value);

OCCA_LFUNC occaType OCCA_RFUNC occaInt64(int64_t value);
OCCA_LFUNC occaType OCCA_RFUNC occaUInt64(uint64_t value);
//  ====================================

//  ---[ Ambiguous Types ]--------------
OCCA_LFUNC occaType OCCA_RFUNC occaChar(char value);
OCCA_LFUNC occaType OCCA_RFUNC occaUChar(unsigned char value);

OCCA_LFUNC occaType OCCA_RFUNC occaShort(short value);
OCCA_LFUNC occaType OCCA_RFUNC occaUShort(unsigned short value);

OCCA_LFUNC occaType OCCA_RFUNC occaInt(int value);
OCCA_LFUNC occaType OCCA_RFUNC occaUInt(unsigned int value);

OCCA_LFUNC occaType OCCA_RFUNC occaLong(long value);
OCCA_LFUNC occaType OCCA_RFUNC occaULong(unsigned long value);

OCCA_LFUNC occaType OCCA_RFUNC occaFloat(float value);
OCCA_LFUNC occaType OCCA_RFUNC occaDouble(double value);

OCCA_LFUNC occaType OCCA_RFUNC occaStruct(void *value, occaUDim_t bytes);
OCCA_LFUNC occaType OCCA_RFUNC occaString(const char *str);
//  ====================================

//  ---[ Properties ]-------------------
OCCA_LFUNC occaObject OCCA_RFUNC occaCreateProperties();
OCCA_LFUNC occaObject OCCA_RFUNC occaCreatePropertiesFromString(const char *c);
OCCA_LFUNC void OCCA_RFUNC occaPropertiesSet(occaProperties properties,
                                             const char *key,
                                             occaType value);
OCCA_LFUNC void OCCA_RFUNC occaPropertiesFree(occaProperties properties);
//  ====================================
//======================================


//---[ Background Device ]--------------
//  |---[ Device ]----------------------
OCCA_LFUNC void OCCA_RFUNC occaSetDevice(occaDevice device);
OCCA_LFUNC void OCCA_RFUNC occaSetDeviceFromInfo(const char *infos);

OCCA_LFUNC void OCCA_RFUNC occaFinish();

OCCA_LFUNC void OCCA_RFUNC occaWaitFor(occaStreamTag tag);

OCCA_LFUNC occaStream OCCA_RFUNC occaCreateStream();
OCCA_LFUNC occaStream OCCA_RFUNC occaGetStream();
OCCA_LFUNC void OCCA_RFUNC occaSetStream(occaStream stream);
OCCA_LFUNC occaStream OCCA_RFUNC occaWrapStream(void *handle_,
                                                const occaProperties props);

OCCA_LFUNC occaStreamTag OCCA_RFUNC occaTagStream();
//  |===================================

//  |---[ Kernel ]----------------------
OCCA_LFUNC occaKernel OCCA_RFUNC occaBuildKernel(const char *filename,
                                                 const char *kernelName,
                                                 const occaProperties props);

OCCA_LFUNC occaKernel OCCA_RFUNC occaBuildKernelFromString(const char *str,
                                                           const char *kernelName,
                                                           const occaProperties props);

OCCA_LFUNC occaKernel OCCA_RFUNC occaBuildKernelFromBinary(const char *filename,
                                                           const char *kernelName);
//  |===================================

//  |---[ Memory ]----------------------
OCCA_LFUNC void OCCA_RFUNC occaMemorySwap(occaMemory a, occaMemory b);

OCCA_LFUNC occaMemory OCCA_RFUNC occaMalloc(const occaUDim_t bytes,
                                            void *src,
                                            occaProperties props);

OCCA_LFUNC void* OCCA_RFUNC occaUvaAlloc(const occaUDim_t bytes,
                                         void *src,
                                         occaProperties props);

OCCA_LFUNC occaMemory OCCA_RFUNC occaWrapMemory(void *handle_,
                                                const occaUDim_t bytes,
                                                occaProperties props);
//  |===================================
//======================================


//---[ Device ]-------------------------
OCCA_LFUNC void OCCA_RFUNC occaPrintModeInfo();

OCCA_LFUNC occaDevice OCCA_RFUNC occaCreateDevice(occaObject info);

OCCA_LFUNC const char* OCCA_RFUNC occaDeviceMode(occaDevice device);

OCCA_LFUNC occaUDim_t OCCA_RFUNC occaDeviceMemorySize(occaDevice device);
OCCA_LFUNC occaUDim_t OCCA_RFUNC occaDeviceMemoryAllocated(occaDevice device);
OCCA_LFUNC occaUDim_t OCCA_RFUNC occaDeviceBytesAllocated(occaDevice device);

OCCA_LFUNC occaKernel OCCA_RFUNC occaDeviceBuildKernel(occaDevice device,
                                                       const char *filename,
                                                       const char *kernelName,
                                                       const occaProperties props);

OCCA_LFUNC occaKernel OCCA_RFUNC occaDeviceBuildKernelFromString(occaDevice device,
                                                                 const char *str,
                                                                 const char *kernelName,
                                                                 const occaProperties props);

OCCA_LFUNC occaKernel OCCA_RFUNC occaDeviceBuildKernelFromBinary(occaDevice device,
                                                                 const char *filename,
                                                                 const char *kernelName);

OCCA_LFUNC occaMemory OCCA_RFUNC occaDeviceMalloc(occaDevice device,
                                                  const occaUDim_t bytes,
                                                  void *src,
                                                  occaProperties props);

OCCA_LFUNC void* OCCA_RFUNC occaDeviceUvaAlloc(occaDevice device,
                                               const occaUDim_t bytes,
                                               void *src,
                                               occaProperties props);

OCCA_LFUNC occaMemory OCCA_RFUNC occaDeviceWrapMemory(occaDevice device,
                                                      void *handle_,
                                                      const occaUDim_t bytes,
                                                      occaProperties props);

OCCA_LFUNC void OCCA_RFUNC occaDeviceFinish(occaDevice device);

OCCA_LFUNC occaStream OCCA_RFUNC occaDeviceCreateStream(occaDevice device);
OCCA_LFUNC occaStream OCCA_RFUNC occaDeviceGetStream(occaDevice device);
OCCA_LFUNC void       OCCA_RFUNC occaDeviceSetStream(occaDevice device, occaStream stream);
OCCA_LFUNC occaStream OCCA_RFUNC occaDeviceWrapStream(occaDevice device,
                                                      void *handle_,
                                                      const occaProperties props);

OCCA_LFUNC occaStreamTag OCCA_RFUNC occaDeviceTagStream(occaDevice device);
OCCA_LFUNC void OCCA_RFUNC occaDeviceWaitForTag(occaDevice device,
                                                occaStreamTag tag);
OCCA_LFUNC double OCCA_RFUNC occaDeviceTimeBetweenTags(occaDevice device,
                                                       occaStreamTag startTag, occaStreamTag endTag);

OCCA_LFUNC void OCCA_RFUNC occaStreamFree(occaStream stream);
OCCA_LFUNC void OCCA_RFUNC occaDeviceFree(occaDevice device);
//======================================


//---[ Kernel ]-------------------------
OCCA_LFUNC const char* OCCA_RFUNC occaKernelMode(occaKernel kernel);
OCCA_LFUNC const char* OCCA_RFUNC occaKernelName(occaKernel kernel);

OCCA_LFUNC occaDevice OCCA_RFUNC occaKernelGetDevice(occaKernel kernel);

OCCA_LFUNC void OCCA_RFUNC occaKernelSetWorkingDims(occaKernel kernel,
                                                    int dims,
                                                    occaDim items,
                                                    occaDim groups);

OCCA_LFUNC void OCCA_RFUNC occaKernelSetAllWorkingDims(occaKernel kernel,
                                                       int dims,
                                                       occaUDim_t itemsX, occaUDim_t itemsY, occaUDim_t itemsZ,
                                                       occaUDim_t groupsX, occaUDim_t groupsY, occaUDim_t groupsZ);

OCCA_LFUNC occaArgumentList OCCA_RFUNC occaCreateArgumentList();

OCCA_LFUNC void OCCA_RFUNC occaArgumentListClear(occaArgumentList list);

OCCA_LFUNC void OCCA_RFUNC occaArgumentListFree(occaArgumentList list);

OCCA_LFUNC void OCCA_RFUNC occaArgumentListAddArg(occaArgumentList list,
                                                  int argPos,
                                                  occaObject type);

OCCA_LFUNC void OCCA_RFUNC occaKernelRun_(occaKernel kernel,
                                          occaArgumentList list);

OCCA_LFUNC void OCCA_RFUNC occaKernelRunN(occaKernel kernel,
                                          const int argc,
                                          occaObject *args);

#include "occa/operators/cKernelOperators.hpp"

OCCA_LFUNC void OCCA_RFUNC occaKernelFree(occaKernel kernel);
//======================================


//---[ Memory ]-------------------------
OCCA_LFUNC const char* OCCA_RFUNC occaMemoryMode(occaMemory memory);

OCCA_LFUNC void* OCCA_RFUNC occaMemoryGetHandle(occaMemory mem,
                                                occaProperties props);

OCCA_LFUNC void OCCA_RFUNC occaMemcpy(void *dest, void *src,
                                      const occaUDim_t bytes,
                                      occaProperties props);

OCCA_LFUNC void OCCA_RFUNC occaCopyMemToMem(occaMemory dest, occaMemory src,
                                            const occaUDim_t bytes,
                                            const occaUDim_t destOffset,
                                            const occaUDim_t srcOffset,
                                            occaProperties props);

OCCA_LFUNC void OCCA_RFUNC occaCopyPtrToMem(occaMemory dest, const void *src,
                                            const occaUDim_t bytes, const occaUDim_t offset,
                                            occaProperties props);

OCCA_LFUNC void OCCA_RFUNC occaCopyMemToPtr(void *dest, occaMemory src,
                                            const occaUDim_t bytes, const occaUDim_t offset,
                                            occaProperties props);

OCCA_LFUNC void OCCA_RFUNC occaMemoryFree(occaMemory memory);
//======================================

//---[ Misc ]---------------------------
OCCA_LFUNC void OCCA_RFUNC occaFree(occaObject obj);
//======================================

OCCA_END_EXTERN_C

#endif
