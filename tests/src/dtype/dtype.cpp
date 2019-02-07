#include <occa.hpp>
#include <occa/tools/testing.hpp>

void testDtype();
void testJsonMethods();

int main(const int argc, const char **argv) {
  testDtype();
  testJsonMethods();

  return 0;
}

void testDtype() {
  ASSERT_EQ(occa::dtypes::float_,
            occa::dtypes::float_);

  occa::dtype fakeFloat("float", occa::dtypes::float_.getBytes());
  ASSERT_EQ(occa::dtypes::float_,
            fakeFloat);

  occa::dtype fakeDouble("double", 0);
  ASSERT_NEQ(occa::dtypes::float_,
             fakeDouble);
  ASSERT_NEQ(occa::dtypes::double_,
             fakeDouble);

  occa::dtype foo1("foo");
  foo1.addField("a", occa::dtypes::double_);

  occa::dtype foo2("foo");
  foo2.addField("a", occa::dtypes::double_);

  occa::dtype foo3("foo");
  foo3.addField("a", occa::dtypes::double_)
    .addField("b", occa::dtypes::double_);

  occa::dtype foo4("foo");
  foo4.addField("b", occa::dtypes::double_)
    .addField("a", occa::dtypes::double_);

  ASSERT_EQ(foo1, foo1);
  ASSERT_EQ(foo1, foo2);
  ASSERT_NEQ(foo1, foo3);
  ASSERT_NEQ(foo1, foo4);
  ASSERT_EQ(foo3, foo4);
}

void testJsonMethods() {
  ASSERT_EQ(occa::dtypes::double_.toJson().toString(),
            occa::json::parse("'double'").toString())

  occa::dtype foo("foo");
  foo.addField("a", occa::dtypes::double_)
    .addField("b", occa::dtypes::double_);

  const std::string fooJsonStr = (
    "{"
    "  foo: ["
    "    { a: 'double' },"
    "    { b: 'double' },"
    "  ],"
    "}"
  );

  occa::json fooJson = occa::json::parse(fooJsonStr);
  ASSERT_EQ(fooJson.toString(),
            foo.toJson().toString());

  occa::dtype foo2 = occa::dtype::fromJson(fooJsonStr);
  ASSERT_EQ(foo, foo2);
}
