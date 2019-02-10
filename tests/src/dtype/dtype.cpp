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
  ASSERT_EQ(occa::dtype::float_,
            occa::dtype::float_);

  occa::dtype_t fakeFloat("float", occa::dtype::float_.bytes());
  ASSERT_NEQ(occa::dtype::float_,
             fakeFloat);

  fakeFloat = occa::dtype::float_;
  ASSERT_EQ(occa::dtype::float_,
            fakeFloat);

  occa::dtype_t fakeDouble("double", 0);
  ASSERT_NEQ(occa::dtype::float_,
             fakeDouble);
  ASSERT_NEQ(occa::dtype::double_,
             fakeDouble);

  occa::dtype_t foo1("foo");
  foo1.addField("a", occa::dtype::double_);

  occa::dtype_t foo2("foo");
  foo2.addField("a", occa::dtype::double_);

  occa::dtype_t foo3("foo");
  foo3.addField("a", occa::dtype::double_)
    .addField("b", occa::dtype::double_);

  occa::dtype_t foo4("foo");
  foo4.addField("b", occa::dtype::double_)
    .addField("a", occa::dtype::double_);

  ASSERT_EQ(foo1, foo1);
  ASSERT_NEQ(foo1, foo2);
  ASSERT_NEQ(foo1, foo3);
  ASSERT_NEQ(foo1, foo4);
  ASSERT_NEQ(foo3, foo4);

  ASSERT_TRUE(foo1.matches(foo1));
  ASSERT_TRUE(foo1.matches(foo2));
  ASSERT_FALSE(foo1.matches(foo3));
  ASSERT_FALSE(foo1.matches(foo4));
  ASSERT_FALSE(foo3.matches(foo4));
}

void testJsonMethods() {
  ASSERT_EQ(occa::dtype::double_.toJson().toString(),
            occa::json::parse("'double'").toString());

  occa::dtype_t foo("foo");
  foo.addField("a", occa::dtype::double_)
    .addField("b", occa::dtype::double_);

  const std::string fooJsonStr = (
    "["
    "  ['a', 'double'],"
    "  ['b', 'double'],"
    "]"
  );

  occa::json fooJson = occa::json::parse(fooJsonStr);
  ASSERT_EQ(fooJson.toString(),
            foo.toJson().toString());

  occa::dtype_t foo2 = occa::dtype_t::fromJson(fooJsonStr);
  ASSERT_NEQ(foo, foo2);
  ASSERT_TRUE(foo.matches(foo2));
}
