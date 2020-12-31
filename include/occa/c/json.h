#ifndef OCCA_C_JSON_HEADER
#define OCCA_C_JSON_HEADER

#include <occa/c/defines.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

occaJson occaCreateJson();


//---[ Global methods ]-----------------
occaJson occaJsonParse(const char *c);

occaJson occaJsonRead(const char *filename);

void occaJsonWrite(occaJson j,
                   const char *filename);

const char* occaJsonDump(occaJson j,
                         const int indent);
//======================================


//---[ Type checks ]--------------------
bool occaJsonIsBoolean(occaJson j);
bool occaJsonIsNumber(occaJson j);
bool occaJsonIsString(occaJson j);
bool occaJsonIsArray(occaJson j);
bool occaJsonIsObject(occaJson j);
//======================================


//---[ Casters ]------------------------
void occaJsonCastToBoolean(occaJson j);
void occaJsonCastToNumber(occaJson j);
void occaJsonCastToString(occaJson j);
void occaJsonCastToArray(occaJson j);
void occaJsonCastToObject(occaJson j);
//======================================


//---[ Getters ]------------------------
bool occaJsonGetBoolean(occaJson j);
occaType occaJsonGetNumber(occaJson j,
                           const int type);
const char* occaJsonGetString(occaJson j);
//======================================


//---[ Object methods ]-----------------
occaType occaJsonObjectGet(occaJson j,
                           const char *key,
                           occaType defaultValue);

void occaJsonObjectSet(occaJson j,
                       const char *key,
                       occaType value);

bool occaJsonObjectHas(occaJson j,
                       const char *key);
//======================================


//---[ Array methods ]------------------
int occaJsonArraySize(occaJson j);

occaType occaJsonArrayGet(occaJson j,
                          const int index);

void occaJsonArrayPush(occaJson j,
                       occaType value);

void occaJsonArrayPop(occaJson j);

void occaJsonArrayInsert(occaJson j,
                         const int index,
                         occaType value);

void occaJsonArrayClear(occaJson j);
//======================================

OCCA_END_EXTERN_C

#endif
