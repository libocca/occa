#ifndef OCCA_C_DTYPE_HEADER
#define OCCA_C_DTYPE_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

//-----[ Methods ]----------------------
OCCA_LFUNC occaDtype OCCA_RFUNC occaCreateDtype(const char *name,
                                                const int bytes);

OCCA_LFUNC occaDtype OCCA_RFUNC occaCreateGlobalDtype(const char *name,
                                                      const int bytes);

OCCA_LFUNC occaDtype OCCA_RFUNC occaCreateDtypeTuple(occaDtype dtype,
                                                     const int size);

OCCA_LFUNC const char* OCCA_RFUNC occaDtypeName(occaDtype dtype);
OCCA_LFUNC int OCCA_RFUNC occaDtypeBytes(occaDtype dtype);

OCCA_LFUNC void OCCA_RFUNC occaDtypeAddField(occaDtype dtype,
                                             const char *field,
                                             occaDtype fieldType);

OCCA_LFUNC int OCCA_RFUNC occaDtypesAreEqual(occaDtype a,
                                             occaDtype b);

OCCA_LFUNC int OCCA_RFUNC occaDtypesMatch(occaDtype a,
                                          occaDtype b);

OCCA_LFUNC occaDtype OCCA_RFUNC occaDtypeFromJson(occaJson json);
OCCA_LFUNC occaDtype OCCA_RFUNC occaDtypeFromJsonString(const char *str);

OCCA_LFUNC occaJson OCCA_RFUNC occaDtypeToJson(occaDtype dtype);
//======================================

//-----[ Builtins ]---------------------
extern occaDtype occaDtypeNone;

extern occaDtype occaDtypeVoid;
extern occaDtype occaDtypeByte;

extern occaDtype occaDtypeBool;
extern occaDtype occaDtypeChar;
extern occaDtype occaDtypeShort;
extern occaDtype occaDtypeInt;
extern occaDtype occaDtypeLong;
extern occaDtype occaDtypeFloat;
extern occaDtype occaDtypeDouble;

extern occaDtype occaDtypeInt8;
extern occaDtype occaDtypeUint8;
extern occaDtype occaDtypeInt16;
extern occaDtype occaDtypeUint16;
extern occaDtype occaDtypeInt32;
extern occaDtype occaDtypeUint32;
extern occaDtype occaDtypeInt64;
extern occaDtype occaDtypeUint64;
extern occaDtype occaDtypeFloat32;
extern occaDtype occaDtypeFloat64;

// OKL Primitives
extern occaDtype occaDtypeUchar2;
extern occaDtype occaDtypeUchar3;
extern occaDtype occaDtypeUchar4;

extern occaDtype occaDtypeChar2;
extern occaDtype occaDtypeChar3;
extern occaDtype occaDtypeChar4;

extern occaDtype occaDtypeUshort2;
extern occaDtype occaDtypeUshort3;
extern occaDtype occaDtypeUshort4;

extern occaDtype occaDtypeShort2;
extern occaDtype occaDtypeShort3;
extern occaDtype occaDtypeShort4;

extern occaDtype occaDtypeUint2;
extern occaDtype occaDtypeUint3;
extern occaDtype occaDtypeUint4;

extern occaDtype occaDtypeInt2;
extern occaDtype occaDtypeInt3;
extern occaDtype occaDtypeInt4;

extern occaDtype occaDtypeUlong2;
extern occaDtype occaDtypeUlong3;
extern occaDtype occaDtypeUlong4;

extern occaDtype occaDtypeLong2;
extern occaDtype occaDtypeLong3;
extern occaDtype occaDtypeLong4;

extern occaDtype occaDtypeFloat2;
extern occaDtype occaDtypeFloat3;
extern occaDtype occaDtypeFloat4;

extern occaDtype occaDtypeDouble2;
extern occaDtype occaDtypeDouble3;
extern occaDtype occaDtypeDouble4;
//======================================

OCCA_END_EXTERN_C

#endif
