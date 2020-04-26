#ifndef OCCA_C_DTYPE_HEADER
#define OCCA_C_DTYPE_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

//-----[ Methods ]----------------------
OCCA_LFUNC occaDtype OCCA_RFUNC occaCreateDtype(const char *name,
                                                const int bytes);

OCCA_LFUNC occaDtype OCCA_RFUNC occaCreateDtypeTuple(occaDtype dtype,
                                                     const int size);

OCCA_LFUNC const char* OCCA_RFUNC occaDtypeName(occaDtype dtype);

OCCA_LFUNC int OCCA_RFUNC occaDtypeBytes(occaDtype dtype);


OCCA_LFUNC void OCCA_RFUNC occaDtypeRegisterType(occaDtype dtype);

OCCA_LFUNC bool OCCA_RFUNC occaDtypeIsRegistered(occaDtype dtype);

OCCA_LFUNC void OCCA_RFUNC occaDtypeAddField(occaDtype dtype,
                                             const char *field,
                                             occaDtype fieldType);

OCCA_LFUNC bool OCCA_RFUNC occaDtypesAreEqual(occaDtype a,
                                              occaDtype b);

OCCA_LFUNC bool OCCA_RFUNC occaDtypesMatch(occaDtype a,
                                           occaDtype b);

OCCA_LFUNC occaDtype OCCA_RFUNC occaDtypeFromJson(occaJson json);
OCCA_LFUNC occaDtype OCCA_RFUNC occaDtypeFromJsonString(const char *str);

OCCA_LFUNC occaJson OCCA_RFUNC occaDtypeToJson(occaDtype dtype);
//======================================

//-----[ Builtins ]---------------------
extern const occaDtype occaDtypeNone;

extern const occaDtype occaDtypeVoid;
extern const occaDtype occaDtypeByte;

extern const occaDtype occaDtypeBool;
extern const occaDtype occaDtypeChar;
extern const occaDtype occaDtypeShort;
extern const occaDtype occaDtypeInt;
extern const occaDtype occaDtypeLong;
extern const occaDtype occaDtypeFloat;
extern const occaDtype occaDtypeDouble;

extern const occaDtype occaDtypeInt8;
extern const occaDtype occaDtypeUint8;
extern const occaDtype occaDtypeInt16;
extern const occaDtype occaDtypeUint16;
extern const occaDtype occaDtypeInt32;
extern const occaDtype occaDtypeUint32;
extern const occaDtype occaDtypeInt64;
extern const occaDtype occaDtypeUint64;

// OKL Primitives
extern const occaDtype occaDtypeUchar2;
extern const occaDtype occaDtypeUchar3;
extern const occaDtype occaDtypeUchar4;

extern const occaDtype occaDtypeChar2;
extern const occaDtype occaDtypeChar3;
extern const occaDtype occaDtypeChar4;

extern const occaDtype occaDtypeUshort2;
extern const occaDtype occaDtypeUshort3;
extern const occaDtype occaDtypeUshort4;

extern const occaDtype occaDtypeShort2;
extern const occaDtype occaDtypeShort3;
extern const occaDtype occaDtypeShort4;

extern const occaDtype occaDtypeUint2;
extern const occaDtype occaDtypeUint3;
extern const occaDtype occaDtypeUint4;

extern const occaDtype occaDtypeInt2;
extern const occaDtype occaDtypeInt3;
extern const occaDtype occaDtypeInt4;

extern const occaDtype occaDtypeUlong2;
extern const occaDtype occaDtypeUlong3;
extern const occaDtype occaDtypeUlong4;

extern const occaDtype occaDtypeLong2;
extern const occaDtype occaDtypeLong3;
extern const occaDtype occaDtypeLong4;

extern const occaDtype occaDtypeFloat2;
extern const occaDtype occaDtypeFloat3;
extern const occaDtype occaDtypeFloat4;

extern const occaDtype occaDtypeDouble2;
extern const occaDtype occaDtypeDouble3;
extern const occaDtype occaDtypeDouble4;
//======================================

OCCA_END_EXTERN_C

#endif
