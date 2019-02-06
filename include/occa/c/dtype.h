#ifndef OCCA_C_DTYPE_HEADER
#define OCCA_C_DTYPE_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

//-----[ Methods ]----------------------
OCCA_LFUNC occaDtype OCCA_RFUNC occaCreateDtype(const char *name,
                                                const int bytes);

OCCA_LFUNC const char* OCCA_RFUNC occaDtypeGetName(occaDtype type);
OCCA_LFUNC int OCCA_RFUNC occaDtypeGetBytes(occaDtype type);

OCCA_LFUNC void OCCA_RFUNC occaDtypeAddField(occaDtype type,
                                             const char *field,
                                             occaDtype fieldType);

OCCA_LFUNC int OCCA_RFUNC occaDtypeIsEqual(occaDtype a,
                                           occaDtype b);

OCCA_LFUNC occaDtype OCCA_RFUNC occaDtypeFromJson(occaJson json);
OCCA_LFUNC occaDtype OCCA_RFUNC occaDtypeFromJsonString(const char *str);

OCCA_LFUNC occaJson OCCA_RFUNC occaDtypeToJson(occaDtype type);
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
//======================================

OCCA_END_EXTERN_C

#endif
