#ifndef OCCA_C_JSON_HEADER
#define OCCA_C_JSON_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

OCCA_LFUNC occaJson OCCA_RFUNC occaCreateJson();


//---[ Global methods ]-----------------
OCCA_LFUNC occaJson OCCA_RFUNC occaJsonParse(const char *c);

OCCA_LFUNC occaJson OCCA_RFUNC occaJsonRead(const char *filename);

OCCA_LFUNC void OCCA_RFUNC occaJsonWrite(occaJson j,
                                         const char *filename);

OCCA_LFUNC const char* OCCA_RFUNC occaJsonDump(occaJson j,
                                               const int indent);
//======================================


//---[ Type checks ]--------------------
OCCA_LFUNC bool OCCA_RFUNC occaJsonIsBoolean(occaJson j);
OCCA_LFUNC bool OCCA_RFUNC occaJsonIsNumber(occaJson j);
OCCA_LFUNC bool OCCA_RFUNC occaJsonIsString(occaJson j);
OCCA_LFUNC bool OCCA_RFUNC occaJsonIsArray(occaJson j);
OCCA_LFUNC bool OCCA_RFUNC occaJsonIsObject(occaJson j);
//======================================


//---[ Casters ]------------------------
OCCA_LFUNC void OCCA_RFUNC occaJsonCastToBoolean(occaJson j);
OCCA_LFUNC void OCCA_RFUNC occaJsonCastToNumber(occaJson j);
OCCA_LFUNC void OCCA_RFUNC occaJsonCastToString(occaJson j);
OCCA_LFUNC void OCCA_RFUNC occaJsonCastToArray(occaJson j);
OCCA_LFUNC void OCCA_RFUNC occaJsonCastToObject(occaJson j);
//======================================


//---[ Getters ]------------------------
OCCA_LFUNC bool OCCA_RFUNC occaJsonGetBoolean(occaJson j);
OCCA_LFUNC occaType OCCA_RFUNC occaJsonGetNumber(occaJson j,
                                                 const int type);
OCCA_LFUNC const char* OCCA_RFUNC occaJsonGetString(occaJson j);
//======================================


//---[ Object methods ]-----------------
OCCA_LFUNC occaType OCCA_RFUNC occaJsonObjectGet(occaJson j,
                                                 const char *key,
                                                 occaType defaultValue);

OCCA_LFUNC void OCCA_RFUNC occaJsonObjectSet(occaJson j,
                                             const char *key,
                                             occaType value);

OCCA_LFUNC bool OCCA_RFUNC occaJsonObjectHas(occaJson j,
                                             const char *key);
//======================================


//---[ Array methods ]------------------
OCCA_LFUNC int OCCA_RFUNC occaJsonArraySize(occaJson j);

OCCA_LFUNC occaType OCCA_RFUNC occaJsonArrayGet(occaJson j,
                                                const int index);

OCCA_LFUNC void OCCA_RFUNC occaJsonArrayPush(occaJson j,
                                             occaType value);

OCCA_LFUNC void OCCA_RFUNC occaJsonArrayPop(occaJson j);

OCCA_LFUNC void OCCA_RFUNC occaJsonArrayInsert(occaJson j,
                                               const int index,
                                               occaType value);

OCCA_LFUNC void OCCA_RFUNC occaJsonArrayClear(occaJson j);
//======================================

OCCA_END_EXTERN_C

#endif
