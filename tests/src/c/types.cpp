#include <occa.hpp>
#include <occa.h>
#include <occa/internal/c/types.hpp>
#include <occa/internal/utils/testing.hpp>

void testNewOccaTypes();
void testCTypeWrappers();

int main(const int argc, const char **argv) {
  testNewOccaTypes();
  testCTypeWrappers();

  return 0;
}

void testNewOccaTypes() {
#define TEST_OCCA_TYPE(VALUE, OCCA_TYPE)        \
  do {                                          \
    occaType v = occa::c::newOccaType(VALUE);   \
    ASSERT_EQ(v.type, OCCA_TYPE);               \
    occaFree(&v);                                \
  } while (0)

  occaType value = occaUndefined;
  ASSERT_TRUE(occaIsUndefined(value));
  ASSERT_FALSE(occaIsDefault(value));

  value = occaDefault;
  ASSERT_FALSE(occaIsUndefined(value));
  ASSERT_TRUE(occaIsDefault(value));

  TEST_OCCA_TYPE((void*) NULL, OCCA_PTR);

  TEST_OCCA_TYPE((bool) true, OCCA_BOOL);
  TEST_OCCA_TYPE((int8_t) 1, OCCA_INT8);
  TEST_OCCA_TYPE((uint8_t) 1, OCCA_UINT8);
  TEST_OCCA_TYPE((int16_t) 1, OCCA_INT16);
  TEST_OCCA_TYPE((uint16_t) 1, OCCA_UINT16);
  TEST_OCCA_TYPE((int32_t) 1, OCCA_INT32);
  TEST_OCCA_TYPE((uint32_t) 1, OCCA_UINT32);
  TEST_OCCA_TYPE((int64_t) 1, OCCA_INT64);
  TEST_OCCA_TYPE((uint64_t) 1, OCCA_UINT64);
  TEST_OCCA_TYPE((float) 1.0, OCCA_FLOAT);
  TEST_OCCA_TYPE((double) 1.0, OCCA_DOUBLE);

  {
    occaType v = occa::c::newOccaType(*(new occa::json()), true);
    ASSERT_EQ(v.type, OCCA_JSON);
    occaFree(&v);
  }
  {
    occa::json props;
    occaType v = occa::c::newOccaType(props, false);
    ASSERT_EQ(v.type, OCCA_JSON);
    occaFree(&v);
  }

  occaJson cProps = (
    occaJsonParse("{a: 1, b: 2}")
  );
  const occa::json &props = occa::c::json(cProps);
  ASSERT_EQ((int) props["a"],
            1);
  ASSERT_EQ((int) props["b"],
            2);

  occaPrintTypeInfo(cProps);

  occaFree(&cProps);

#undef TEST_OCCA_TYPE
}

template <class TM>
int getOccaType(bool isUnsigned) {
  switch (sizeof(TM)) {
  case 1: return isUnsigned ? OCCA_UINT8 : OCCA_INT8;
  case 2: return isUnsigned ? OCCA_UINT16 : OCCA_INT16;
  case 4: return isUnsigned ? OCCA_UINT32 : OCCA_INT32;
  case 8: return isUnsigned ? OCCA_UINT64 : OCCA_INT64;
  }
  return OCCA_UNDEFINED;
}

void testCTypeWrappers() {
#define TEST_OCCA_C_TYPE(VALUE, OCCA_TYPE)      \
  do {                                          \
    occaType value = VALUE;                     \
    occaPrintTypeInfo(value);                   \
    ASSERT_EQ(value.type, OCCA_TYPE);           \
} while (0)

  TEST_OCCA_C_TYPE(occaPtr(NULL), OCCA_PTR);

  TEST_OCCA_C_TYPE(occaBool(true), OCCA_BOOL);
  TEST_OCCA_C_TYPE(occaInt8(1), OCCA_INT8);
  TEST_OCCA_C_TYPE(occaUInt8(1), OCCA_UINT8);
  TEST_OCCA_C_TYPE(occaInt16(1), OCCA_INT16);
  TEST_OCCA_C_TYPE(occaUInt16(1), OCCA_UINT16);
  TEST_OCCA_C_TYPE(occaInt32(1), OCCA_INT32);
  TEST_OCCA_C_TYPE(occaUInt32(1), OCCA_UINT32);
  TEST_OCCA_C_TYPE(occaInt64(1), OCCA_INT64);
  TEST_OCCA_C_TYPE(occaUInt64(1), OCCA_UINT64);

  TEST_OCCA_C_TYPE(occaChar(1), getOccaType<char>(false));
  TEST_OCCA_C_TYPE(occaUChar(1), getOccaType<unsigned char>(true));
  TEST_OCCA_C_TYPE(occaShort(1), getOccaType<short>(false));
  TEST_OCCA_C_TYPE(occaUShort(1), getOccaType<unsigned short>(true));
  TEST_OCCA_C_TYPE(occaInt(1), getOccaType<int>(false));
  TEST_OCCA_C_TYPE(occaUInt(1), getOccaType<unsigned int>(true));
  TEST_OCCA_C_TYPE(occaLong(1), getOccaType<long>(false));
  TEST_OCCA_C_TYPE(occaULong(1), getOccaType<unsigned long>(true));

  TEST_OCCA_C_TYPE(occaFloat(1.0), OCCA_FLOAT);
  TEST_OCCA_C_TYPE(occaDouble(1.0), OCCA_DOUBLE);

  TEST_OCCA_C_TYPE(occaStruct(NULL, 0), OCCA_STRUCT);

  std::string str = "123";
  TEST_OCCA_C_TYPE(occaString(str.c_str()), OCCA_STRING);

#undef TEST_OCCA_C_TYPE
}
