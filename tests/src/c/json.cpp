#define OCCA_DISABLE_VARIADIC_MACROS

#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <occa.hpp>
#include <occa.h>
#include <occa/internal/c/types.hpp>
#include <occa/internal/utils/testing.hpp>

void testTypeChecking();
void testTypes();
void testArray();
void testBadType();
void testKeyMiss();
void testSerialization();
void testCasting();

occaJson cJson;

int main(const int argc, const char **argv) {
  srand(time(NULL));

  cJson = occaCreateJson();

  testTypeChecking();
  testTypes();
  testArray();
  testKeyMiss();
  testSerialization();
  testBadType();
  testCasting();

  occaFree(&cJson);

  return 0;
}

void testTypeChecking() {
  occaJson cJson2 = occaJsonParse(
    "{"
    "  bool: true,"
    "  number: 1,"
    "  string: 'string',"
    "  array: [],"
    "}"
  );

  ASSERT_TRUE(
    occaJsonIsObject(cJson2)
  );

  ASSERT_TRUE(
    occaJsonIsBoolean(
      occaJsonObjectGet(cJson2, "bool", occaUndefined)
    )
  );

  ASSERT_TRUE(
    occaJsonIsNumber(
      occaJsonObjectGet(cJson2, "number", occaUndefined)
    )
  );

  ASSERT_TRUE(
    occaJsonIsString(
      occaJsonObjectGet(cJson2, "string", occaUndefined)
    )
  );

  ASSERT_TRUE(
    occaJsonIsArray(
      occaJsonObjectGet(cJson2, "array", occaUndefined)
    )
  );

  ASSERT_FALSE(occaJsonObjectHas(cJson2, "undefined"));
  ASSERT_TRUE(occaJsonObjectHas(cJson2, "bool"));
  ASSERT_TRUE(occaJsonObjectHas(cJson2, "number"));
  ASSERT_TRUE(occaJsonObjectHas(cJson2, "string"));
  ASSERT_TRUE(occaJsonObjectHas(cJson2, "array"));

  occaFree(&cJson2);
}

void testTypes() {
  bool boolValue    = (bool) rand() % 2;
  int8_t i8Value    = (int8_t) rand();
  uint8_t u8Value   = (uint8_t) rand();
  int16_t i16Value  = (int16_t) rand();
  uint16_t u16Value = (uint16_t) rand();
  int32_t i32Value  = (int32_t) rand();
  uint32_t u32Value = (uint32_t) rand();
  int64_t i64Value  = (int64_t) rand();
  uint64_t u64Value = (uint64_t) rand();
  std::string stringValue = occa::toString(rand());

  occaType bool_  = occaBool(boolValue);
  occaType i8     = occaInt8(i8Value);
  occaType u8     = occaUInt8(u8Value);
  occaType i16    = occaInt16(i16Value);
  occaType u16    = occaUInt16(u16Value);
  occaType i32    = occaInt32(i32Value);
  occaType u32    = occaUInt32(u32Value);
  occaType i64    = occaInt64(i64Value);
  occaType u64    = occaUInt64(u64Value);
  occaType string = occaString(stringValue.c_str());

  occaJsonObjectSet(cJson, "bool"    , bool_);
  occaJsonObjectSet(cJson, "int8_t"  , i8);
  occaJsonObjectSet(cJson, "uint8_t" , u8);
  occaJsonObjectSet(cJson, "int16_t" , i16);
  occaJsonObjectSet(cJson, "uint16_t", u16);
  occaJsonObjectSet(cJson, "int32_t" , i32);
  occaJsonObjectSet(cJson, "uint32_t", u32);
  occaJsonObjectSet(cJson, "int64_t" , i64);
  occaJsonObjectSet(cJson, "uint64_t", u64);
  occaJsonObjectSet(cJson, "string"  , string);

  occaType undef  = occaJsonObjectGet(cJson, "undefined", occaUndefined);
  ASSERT_NEQ(undef.type, OCCA_JSON);

  occaType bool_2   = occaJsonObjectGet(cJson, "bool"    , occaUndefined);
  occaType i8_2     = occaJsonObjectGet(cJson, "int8_t"  , occaUndefined);
  occaType u8_2     = occaJsonObjectGet(cJson, "uint8_t" , occaUndefined);
  occaType i16_2    = occaJsonObjectGet(cJson, "int16_t" , occaUndefined);
  occaType u16_2    = occaJsonObjectGet(cJson, "uint16_t", occaUndefined);
  occaType i32_2    = occaJsonObjectGet(cJson, "int32_t" , occaUndefined);
  occaType u32_2    = occaJsonObjectGet(cJson, "uint32_t", occaUndefined);
  occaType i64_2    = occaJsonObjectGet(cJson, "int64_t" , occaUndefined);
  occaType u64_2    = occaJsonObjectGet(cJson, "uint64_t", occaUndefined);
  occaType string_2 = occaJsonObjectGet(cJson, "string"  , occaUndefined);

  ASSERT_EQ(bool_2.type  , OCCA_JSON);
  ASSERT_EQ(i8_2.type    , OCCA_JSON);
  ASSERT_EQ(u8_2.type    , OCCA_JSON);
  ASSERT_EQ(i16_2.type   , OCCA_JSON);
  ASSERT_EQ(u16_2.type   , OCCA_JSON);
  ASSERT_EQ(i32_2.type   , OCCA_JSON);
  ASSERT_EQ(u32_2.type   , OCCA_JSON);
  ASSERT_EQ(i64_2.type   , OCCA_JSON);
  ASSERT_EQ(u64_2.type   , OCCA_JSON);
  ASSERT_EQ(string_2.type, OCCA_JSON);

  ASSERT_EQ(boolValue, (bool) occaJsonGetBoolean(bool_2));
  ASSERT_EQ(stringValue.c_str(), occaJsonGetString(string_2));

  occaType i8_3   = occaJsonGetNumber(i8_2, OCCA_INT8);
  occaType u8_3   = occaJsonGetNumber(u8_2, OCCA_UINT8);
  occaType i16_3  = occaJsonGetNumber(i16_2, OCCA_INT16);
  occaType u16_3  = occaJsonGetNumber(u16_2, OCCA_UINT16);
  occaType i32_3  = occaJsonGetNumber(i32_2, OCCA_INT32);
  occaType u32_3  = occaJsonGetNumber(u32_2, OCCA_UINT32);
  occaType i64_3  = occaJsonGetNumber(i64_2, OCCA_INT64);
  occaType u64_3  = occaJsonGetNumber(u64_2, OCCA_UINT64);

  ASSERT_EQ((int) i8_3.value.int8_, (int) i8Value);
  ASSERT_EQ((int) u8_3.value.uint8_, (int) u8Value);
  ASSERT_EQ(i16_3.value.int16_, i16Value);
  ASSERT_EQ(u16_3.value.uint16_, u16Value);
  ASSERT_EQ(i32_3.value.int32_, i32Value);
  ASSERT_EQ(u32_3.value.uint32_, u32Value);
  ASSERT_EQ(i64_3.value.int64_, i64Value);
  ASSERT_EQ(u64_3.value.uint64_, u64Value);

  // NULL
  occaJsonObjectSet(cJson, "null", occaNull);
  occaType nullValue = occaJsonObjectGet(cJson, "null", occaUndefined);
  ASSERT_EQ(nullValue.type, OCCA_NULL);
  ASSERT_EQ(nullValue.value.ptr, (void*) NULL);

  // Nested props
  occaJson cJson2 = occaJsonParse(
    "{ prop: { value: 1 } }"
  );
  occaType propValue = occaJsonObjectGet(cJson2, "prop", occaUndefined);
  ASSERT_EQ(propValue.type, OCCA_JSON);
  ASSERT_TRUE(occaJsonIsObject(propValue));
  ASSERT_TRUE(occaJsonObjectHas(propValue, "value"));

  occaFree(&cJson2);
}

void testArray() {
  occaJson array = occaJsonParse(
    "[true, 1, 'string', [], {}]"
  );

  ASSERT_EQ(occaJsonArraySize(array), 5);

  ASSERT_TRUE(
    occaJsonIsBoolean(
      occaJsonArrayGet(array, 0)
    )
  );

  ASSERT_TRUE(
    occaJsonIsNumber(
      occaJsonArrayGet(array, 1)
    )
  );

  ASSERT_TRUE(
    occaJsonIsString(
      occaJsonArrayGet(array, 2)
    )
  );

  ASSERT_TRUE(
    occaJsonIsArray(
      occaJsonArrayGet(array, 3)
    )
  );

  ASSERT_TRUE(
    occaJsonIsObject(
      occaJsonArrayGet(array, 4)
    )
  );

  occaJsonArrayPush(array, occaInt(1));
  ASSERT_EQ(occaJsonArraySize(array), 6);

  occaJsonArrayPop(array);
  ASSERT_EQ(occaJsonArraySize(array), 5);


  occaJsonArrayInsert(array, 0, occaInt(1));
  ASSERT_EQ(occaJsonArraySize(array), 6);

  ASSERT_TRUE(
    occaJsonIsNumber(
      occaJsonArrayGet(array, 0)
    )
  );

  occaJsonArrayClear(array);
  ASSERT_EQ(occaJsonArraySize(array), 0);

  occaFree(&array);
}

void testBadType() {
  ASSERT_THROW(
    occaJsonObjectSet(cJson, "ptr", occaPtr((void*) 10));
  );

  ASSERT_THROW(
    occaJsonObjectSet(cJson, "device", occaGetDevice());
  );
}

void testKeyMiss() {
  // Test get miss
  ASSERT_FALSE(occaJsonObjectHas(cJson, "foobar"));

  occaType foobar = occaJsonObjectGet(cJson, "foobar", occaUndefined);
  ASSERT_TRUE(occaIsUndefined(foobar));

  foobar = occaJsonObjectGet(cJson, "foobar", occaInt32(2));
  ASSERT_EQ(foobar.type,
            OCCA_INT32);
  ASSERT_EQ(foobar.value.int32_,
            2);

  // Set 'foobar'
  std::string hi = "hi";
  occaJsonObjectSet(cJson, "foobar", occaString(hi.c_str()));

  // Test success
  ASSERT_TRUE(occaJsonObjectHas(cJson, "foobar"));

  foobar = occaJsonObjectGet(cJson, "foobar", occaInt32(2));
  ASSERT_TRUE(occaJsonIsString(foobar));
  ASSERT_EQ(hi, occaJsonGetString(foobar));
}

void testSerialization() {
  occa::json &props = occa::c::json(cJson);

  const std::string propStr = (std::string) props;
  occaJson cJson2 = occaJsonParse(propStr.c_str());
  occa::json &props2 = occa::c::json(cJson2);

  ASSERT_EQ(props,
            props2);

  occaFree(&cJson2);
}

void testCasting() {
  occaJson cJson2 = occaCreateJson();

  occaJsonCastToBoolean(cJson2);
  ASSERT_TRUE(occaJsonIsBoolean(cJson2));

  occaJsonCastToNumber(cJson2);
  ASSERT_TRUE(occaJsonIsNumber(cJson2));

  occaJsonCastToString(cJson2);
  ASSERT_TRUE(occaJsonIsString(cJson2));

  occaJsonCastToArray(cJson2);
  ASSERT_TRUE(occaJsonIsArray(cJson2));

  occaJsonCastToObject(cJson2);
  ASSERT_TRUE(occaJsonIsObject(cJson2));

  occaFree(&cJson2);
}
