#define OCCA_DISABLE_VARIADIC_MACROS

#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <occa.h>
#include <occa/internal/utils/testing.hpp>

void testDtype();
void testJsonMethods();

int main(const int argc, const char **argv) {
  testDtype();
  testJsonMethods();

  return 0;
}

void testDtype() {
  ASSERT_TRUE(
    occaDtypesAreEqual(occaDtypeFloat,
                       occaDtypeFloat)
  );

  occaDtype fakeFloat = occaCreateDtype("float",
                                        occaDtypeBytes(occaDtypeFloat));
  ASSERT_FALSE(
    occaDtypesAreEqual(occaDtypeFloat,
                       fakeFloat)
  );

  occaDtype fakeDouble = occaCreateDtype("double", 0);
  ASSERT_FALSE(
    occaDtypesAreEqual(occaDtypeFloat,
                       fakeDouble)
  );
  ASSERT_FALSE(
    occaDtypesAreEqual(occaDtypeDouble,
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
    occaDtypesAreEqual(foo1, foo1)
  );
  ASSERT_FALSE(
    occaDtypesAreEqual(foo1, foo2)
  );
  ASSERT_FALSE(
    occaDtypesAreEqual(foo1, foo3)
  );
  ASSERT_FALSE(
    occaDtypesAreEqual(foo1, foo4)
  );
  ASSERT_FALSE(
    occaDtypesAreEqual(foo3, foo4)
  );

  ASSERT_TRUE(
    occaDtypesMatch(foo1, foo1)
  );
  ASSERT_TRUE(
    occaDtypesMatch(foo1, foo2)
  );
  ASSERT_FALSE(
    occaDtypesMatch(foo1, foo3)
  );
  ASSERT_FALSE(
    occaDtypesMatch(foo1, foo4)
  );
  ASSERT_FALSE(
    occaDtypesMatch(foo3, foo4)
  );

  occaFree(&fakeFloat);
  occaFree(&fakeDouble);
  occaFree(&foo1);
  occaFree(&foo2);
  occaFree(&foo3);
  occaFree(&foo4);
}

void testJsonMethods() {
  occaJson doubleJson = occaDtypeToJson(occaDtypeDouble);
  const char *doubleJsonStr = occaJsonDump(doubleJson, 0);

  occaJson rawDoubleJson = occaJsonParse("{ type: 'builtin', name: 'double' }");
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
    "  type: 'struct',"
    "  fields: ["
    "    { name: 'a', dtype: { type: 'builtin', name: 'double' } },"
    "    { name: 'b', dtype: { type: 'builtin', name: 'double' } },"
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
  ASSERT_FALSE(
    occaDtypesAreEqual(foo, foo2)
  );
  ASSERT_TRUE(
    occaDtypesMatch(foo, foo2)
  );

  occaDtype foo3 = occaDtypeFromJsonString(baseFooJsonStr.c_str());
  ASSERT_FALSE(
    occaDtypesAreEqual(foo, foo3)
  );
  ASSERT_FALSE(
    occaDtypesAreEqual(foo2, foo3)
  );
  ASSERT_TRUE(
    occaDtypesMatch(foo, foo3)
  );
  ASSERT_TRUE(
    occaDtypesMatch(foo2, foo3)
  );

  ::free((void*) doubleJsonStr);
  ::free((void*) rawDoubleJsonStr);
  ::free((void*) fooJsonStr);
  ::free((void*) rawFooJsonStr);

  occaFree(&doubleJson);
  occaFree(&rawDoubleJson);
  occaFree(&foo);
  occaFree(&fooJson);
  occaFree(&rawFooJson);
  occaFree(&foo2);
  occaFree(&foo3);
}
