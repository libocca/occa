#define OCCA_DISABLE_VARIADIC_MACROS

#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <occa.hpp>
#include <occa.h>
#include <occa/c/types.hpp>
#include <occa/tools/testing.hpp>

void testTypes();
void testBadType();
void testKeyMiss();
void testSerialization();
void testCasting();

occaProperties cProps;

int main(const int argc, const char **argv) {
  srand(time(NULL));

  cProps = occaCreateProperties();

  testTypes();
  testKeyMiss();
  testSerialization();
  testBadType();

  occaFree(&cProps);

  return 0;
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

  occaPropertiesSet(cProps, "bool"    , bool_);
  occaPropertiesSet(cProps, "int8_t"  , i8);
  occaPropertiesSet(cProps, "uint8_t" , u8);
  occaPropertiesSet(cProps, "int16_t" , i16);
  occaPropertiesSet(cProps, "uint16_t", u16);
  occaPropertiesSet(cProps, "int32_t" , i32);
  occaPropertiesSet(cProps, "uint32_t", u32);
  occaPropertiesSet(cProps, "int64_t" , i64);
  occaPropertiesSet(cProps, "uint64_t", u64);
  occaPropertiesSet(cProps, "string"  , string);

  occaType undef  = occaPropertiesGet(cProps, "undefined", occaUndefined);
  ASSERT_NEQ(undef.type, OCCA_JSON);

  occaType bool_2   = occaPropertiesGet(cProps, "bool"    , occaUndefined);
  occaType i8_2     = occaPropertiesGet(cProps, "int8_t"  , occaUndefined);
  occaType u8_2     = occaPropertiesGet(cProps, "uint8_t" , occaUndefined);
  occaType i16_2    = occaPropertiesGet(cProps, "int16_t" , occaUndefined);
  occaType u16_2    = occaPropertiesGet(cProps, "uint16_t", occaUndefined);
  occaType i32_2    = occaPropertiesGet(cProps, "int32_t" , occaUndefined);
  occaType u32_2    = occaPropertiesGet(cProps, "uint32_t", occaUndefined);
  occaType i64_2    = occaPropertiesGet(cProps, "int64_t" , occaUndefined);
  occaType u64_2    = occaPropertiesGet(cProps, "uint64_t", occaUndefined);
  occaType string_2 = occaPropertiesGet(cProps, "string"  , occaUndefined);

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
  occaPropertiesSet(cProps, "null", occaNull);
  occaType nullValue = occaPropertiesGet(cProps, "null", occaUndefined);
  ASSERT_EQ(nullValue.type, OCCA_NULL);
  ASSERT_EQ(nullValue.value.ptr, (void*) NULL);

  // Nested props
  occaProperties cProps2 = occaCreatePropertiesFromString(
    "prop: { value: 1 }"
  );
  occaType propValue = occaPropertiesGet(cProps2, "prop", occaUndefined);
  ASSERT_EQ(propValue.type, OCCA_JSON);
  ASSERT_TRUE(occaJsonIsObject(propValue));
  ASSERT_TRUE(occaJsonObjectHas(propValue, "value"));

  occaFree(&cProps2);
}

void testBadType() {
  ASSERT_THROW(
    occaPropertiesSet(cProps, "ptr", occaPtr((void*) 10));
  );

  ASSERT_THROW(
    occaPropertiesSet(cProps, "device", occaGetDevice());
  );
}

void testKeyMiss() {
  // Test get miss
  ASSERT_FALSE(occaPropertiesHas(cProps, "foobar"));

  occaType foobar = occaPropertiesGet(cProps, "foobar", occaUndefined);
  ASSERT_TRUE(occaIsUndefined(foobar));

  foobar = occaPropertiesGet(cProps, "foobar", occaInt32(2));
  ASSERT_EQ(foobar.type,
            OCCA_INT32);
  ASSERT_EQ(foobar.value.int32_,
            2);

  // Set 'foobar'
  std::string hi = "hi";
  occaPropertiesSet(cProps, "foobar", occaString(hi.c_str()));

  // Test success
  ASSERT_TRUE(occaPropertiesHas(cProps, "foobar"));

  foobar = occaPropertiesGet(cProps, "foobar", occaInt32(2));
  ASSERT_TRUE(occaJsonIsString(foobar));
  ASSERT_EQ(hi, occaJsonGetString(foobar));
}

void testSerialization() {
  occa::properties &props = occa::c::properties(cProps);

  const std::string propStr = (std::string) props;
  occaProperties cProps2 = occaCreatePropertiesFromString(propStr.c_str());
  occa::properties &props2 = occa::c::properties(cProps2);

  ASSERT_EQ(props,
            props2);

  occaFree(&cProps2);
}

void testCasting() {
  occaProperties cProps2 = occaCreateJson();

  occaJsonCastToBoolean(cProps2);
  ASSERT_TRUE(occaJsonIsBoolean(cProps2));

  occaJsonCastToNumber(cProps2);
  ASSERT_TRUE(occaJsonIsNumber(cProps2));

  occaJsonCastToString(cProps2);
  ASSERT_TRUE(occaJsonIsString(cProps2));

  occaJsonCastToArray(cProps2);
  ASSERT_TRUE(occaJsonIsArray(cProps2));

  occaJsonCastToObject(cProps2);
  ASSERT_TRUE(occaJsonIsObject(cProps2));

  occaFree(&cProps2);
}
