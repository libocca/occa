#define OCCA_DISABLE_VARIADIC_MACROS

#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <occa.h>
#include <occa/tools/testing.hpp>

void testDtype();
void testJsonMethods();

int main(const int argc, const char **argv) {
  testDtype();
  testJsonMethods();

  return 0;
}

void testDtype() {
  ASSERT_TRUE(
    occaDtypeIsEqual(occaDtypeFloat,
                     occaDtypeFloat)
  );

  occaDtype fakeFloat = occaCreateDtype("float",
                                        occaDtypeGetBytes(occaDtypeFloat));
  ASSERT_TRUE(
    occaDtypeIsEqual(occaDtypeFloat,
                     fakeFloat)
  );

  occaDtype fakeDouble = occaCreateDtype("double", 0);
  ASSERT_FALSE(
    occaDtypeIsEqual(occaDtypeFloat,
                     fakeDouble)
  );
  ASSERT_FALSE(
    occaDtypeIsEqual(occaDtypeDouble,
                     fakeDouble)
  );

  occaDtype foo1 = occaCreateDtype("foo", 0);
  occaDtypeAddField(foo1,
                    "a", occaDtypeDouble);

  occaDtype foo2 = occaCreateDtype("foo", 0);
  occaDtypeAddField(foo2,
                    "a", occaDtypeDouble);

  occaDtype foo3 = occaCreateDtype("foo", 0);
  occaDtypeAddField(foo3,
                    "a", occaDtypeDouble);
  occaDtypeAddField(foo3,
                    "b", occaDtypeDouble);

  occaDtype foo4 = occaCreateDtype("foo", 0);
  occaDtypeAddField(foo4,
                    "b", occaDtypeDouble);
  occaDtypeAddField(foo4,
                    "a", occaDtypeDouble);

  ASSERT_TRUE(
    occaDtypeIsEqual(foo1, foo1)
  );
  ASSERT_TRUE(
    occaDtypeIsEqual(foo1, foo2)
  );
  ASSERT_FALSE(
    occaDtypeIsEqual(foo1, foo3)
  );
  ASSERT_FALSE(
    occaDtypeIsEqual(foo1, foo4)
  );
  ASSERT_TRUE(
    occaDtypeIsEqual(foo3, foo4)
  );

  occaFree(fakeFloat);
  occaFree(fakeDouble);
  occaFree(foo1);
  occaFree(foo2);
  occaFree(foo3);
  occaFree(foo4);
}

void testJsonMethods() {
  occaJson doubleJson = occaDtypeToJson(occaDtypeDouble);
  const char *doubleJsonStr = occaJsonDump(doubleJson, 0);

  occaJson rawDoubleJson = occaJsonParse("'double'");
  const char *rawDoubleJsonStr = occaJsonDump(rawDoubleJson, 0);

  ASSERT_EQ(doubleJsonStr,
            rawDoubleJsonStr);

  occaDtype foo = occaCreateDtype("foo", 0);
  occaDtypeAddField(foo,
                    "a", occaDtypeDouble);
  occaDtypeAddField(foo,
                    "b", occaDtypeDouble);

  const std::string baseFooJsonStr = (
    "{"
    "  foo: ["
    "    { a: 'double' },"
    "    { b: 'double' },"
    "  ],"
    "}"
  );

  occaJson fooJson = occaDtypeToJson(foo);
  const char *fooJsonStr = occaJsonDump(fooJson, 0);

  occaJson rawFooJson = occaJsonParse(baseFooJsonStr.c_str());
  const char *rawFooJsonStr = occaJsonDump(rawFooJson, 0);

  ASSERT_EQ(fooJsonStr,
            rawFooJsonStr);

  occaDtype foo2 = occaDtypeFromJson(fooJson);
  ASSERT_TRUE(
    occaDtypeIsEqual(foo, foo2)
  );

  occaDtype foo3 = occaDtypeFromJsonString(baseFooJsonStr.c_str());
  ASSERT_TRUE(
    occaDtypeIsEqual(foo, foo3)
  );
  ASSERT_TRUE(
    occaDtypeIsEqual(foo2, foo3)
  );

  ::free((void*) doubleJsonStr);
  ::free((void*) rawDoubleJsonStr);
  ::free((void*) fooJsonStr);
  ::free((void*) rawFooJsonStr);

  occaFree(doubleJson);
  occaFree(rawDoubleJson);
  occaFree(foo);
  occaFree(fooJson);
  occaFree(rawFooJson);
  occaFree(foo2);
  occaFree(foo3);
}
