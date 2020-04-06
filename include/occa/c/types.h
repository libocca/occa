#ifndef OCCA_C_TYPES_HEADER
#define OCCA_C_TYPES_HEADER

#include <stdint.h>
#include <stdlib.h>

#include <occa/c/defines.h>

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
  char needsFree;

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
typedef occaType occaKernelBuilder;
typedef occaType occaMemory;
typedef occaType occaStream;
typedef occaType occaStreamTag;

typedef occaType occaDtype;
typedef occaType occaScope;
typedef occaType occaJson;
typedef occaType occaProperties;

//---[ Type Flags ]---------------------
extern const int OCCA_UNDEFINED;
extern const int OCCA_DEFAULT;
extern const int OCCA_NULL;

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
extern const int OCCA_KERNELBUILDER;
extern const int OCCA_MEMORY;
extern const int OCCA_STREAM;
extern const int OCCA_STREAMTAG;

extern const int OCCA_DTYPE;
extern const int OCCA_SCOPE;
extern const int OCCA_JSON;
extern const int OCCA_PROPERTIES;
//======================================

//---[ Globals & Flags ]----------------
extern const occaType occaNull;
extern const occaType occaUndefined;
extern const occaType occaDefault;
extern const occaUDim_t occaAllBytes;
//======================================

//-----[ Known Types ]------------------
OCCA_LFUNC int OCCA_RFUNC occaIsUndefined(occaType value);
OCCA_LFUNC int OCCA_RFUNC occaIsDefault(occaType value);

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

OCCA_LFUNC occaType OCCA_RFUNC occaStruct(const void *value,
                                          occaUDim_t bytes);

OCCA_LFUNC occaType OCCA_RFUNC occaString(const char *str);
//======================================

OCCA_LFUNC void OCCA_RFUNC occaFree(occaType value);

OCCA_END_EXTERN_C

#endif
