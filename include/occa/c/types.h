/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
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

#ifndef OCCA_C_TYPES_HEADER
#define OCCA_C_TYPES_HEADER

#include "stdint.h"
#include "stdlib.h"

#include "occa/c/defines.h"

OCCA_START_EXTERN_C

typedef int64_t  occaDim_t;
typedef uint64_t occaUDim_t;

typedef struct {
  occaUDim_t x, y, z;
} occaDim;

typedef struct {
  int magicHeader;
  int type;
  occaUDim_t bytes;

  union {
    uint8_t  uint8_;
    uint16_t uint16_;
    uint32_t uint32_;
    uint64_t uint64_;

    int8_t  int8_;
    int16_t int16_;
    int32_t int32_;
    int64_t int64_;

    float  float_;
    double double_;

    char* ptr;
  } value;
} occaType;

typedef occaType occaDevice;
typedef occaType occaKernel;
typedef occaType occaMemory;
typedef occaType occaProperties;

typedef struct {
  occaDevice device;
  void *handle;
} occaStream;

typedef struct {
  double tagTime;
  void *handle;
} occaStreamTag;

//---[ Type Flags ]---------------------
extern const int OCCA_PTR;

extern const int OCCA_BOOL;
extern const int OCCA_INT8;
extern const int OCCA_UINT8;
extern const int OCCA_INT16;
extern const int OCCA_UINT16;
extern const int OCCA_INT32;
extern const int OCCA_UINT32;
extern const int OCCA_INT64;
extern const int OCCA_UINT64;
extern const int OCCA_FLOAT;
extern const int OCCA_DOUBLE;

extern const int OCCA_STRUCT;
extern const int OCCA_STRING;

extern const int OCCA_DEVICE;
extern const int OCCA_KERNEL;
extern const int OCCA_MEMORY;
extern const int OCCA_PROPERTIES;
//======================================

//---[ Globals & Flags ]----------------
extern const occaType occaDefault;
extern const occaUDim_t occaAllBytes;
//======================================

//-----[ Known Types ]------------------
OCCA_LFUNC occaType OCCA_RFUNC occaPtr(void *value);

OCCA_LFUNC occaType OCCA_RFUNC occaBool(int value);

OCCA_LFUNC occaType OCCA_RFUNC occaInt8(int8_t value);
OCCA_LFUNC occaType OCCA_RFUNC occaUInt8(uint8_t value);

OCCA_LFUNC occaType OCCA_RFUNC occaInt16(int16_t value);
OCCA_LFUNC occaType OCCA_RFUNC occaUInt16(uint16_t value);

OCCA_LFUNC occaType OCCA_RFUNC occaInt32(int32_t value);
OCCA_LFUNC occaType OCCA_RFUNC occaUInt32(uint32_t value);

OCCA_LFUNC occaType OCCA_RFUNC occaInt64(int64_t value);
OCCA_LFUNC occaType OCCA_RFUNC occaUInt64(uint64_t value);
//======================================

//-----[ Ambiguous Types ]--------------
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

OCCA_LFUNC occaType OCCA_RFUNC occaStruct(void *value,
                                          occaUDim_t bytes);

OCCA_LFUNC occaType OCCA_RFUNC occaString(const char *str);
//======================================

OCCA_LFUNC void OCCA_RFUNC occaFree(occaType value);
OCCA_LFUNC void OCCA_RFUNC occaFreeStream(occaStream stream);

OCCA_END_EXTERN_C

#endif
