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
OCCA_LFUNC int OCCA_RFUNC occaJsonIsBoolean(occaJson j);
OCCA_LFUNC int OCCA_RFUNC occaJsonIsNumber(occaJson j);
OCCA_LFUNC int OCCA_RFUNC occaJsonIsString(occaJson j);
OCCA_LFUNC int OCCA_RFUNC occaJsonIsArray(occaJson j);
OCCA_LFUNC int OCCA_RFUNC occaJsonIsObject(occaJson j);
//======================================


//---[ Getters ]------------------------
OCCA_LFUNC int OCCA_RFUNC occaJsonGetBoolean(occaJson j);
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

OCCA_LFUNC int OCCA_RFUNC occaJsonObjectHas(occaJson j,
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
