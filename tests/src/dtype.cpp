#include <occa.hpp>
#include <occa/internal/utils/testing.hpp>

void testDtype();
void testCasting();
void testGet();
void testJsonMethods();

int main(const int argc, const char **argv) {
  testDtype();
  testCasting();
  testGet();
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

void testCasting() {
  occa::dtype_t foo1("foo");
  foo1.addField("a", occa::dtype::double_)
    .addField("b", occa::dtype::double_);

  occa::dtype_t foo2("foo");
  foo2.addField("b", occa::dtype::double_)
    .addField("a", occa::dtype::double_);

  ASSERT_NEQ(foo1, foo2);
  ASSERT_FALSE(foo1.matches(foo2));
  ASSERT_TRUE(foo1.canBeCastedTo(foo2));
  ASSERT_TRUE(foo2.canBeCastedTo(foo1));

  // double  <---> double2
  ASSERT_NEQ(occa::dtype::double_,
             occa::dtype::double2);
  ASSERT_FALSE(
    occa::dtype::double_.matches(occa::dtype::double2)
  );
  ASSERT_TRUE(
    occa::dtype::double_.canBeCastedTo(occa::dtype::double2)
  );
  ASSERT_TRUE(
    occa::dtype::double2.canBeCastedTo(occa::dtype::double_)
  );

  // double  <---> double3
  // double2 <-!-> dobule3
  ASSERT_TRUE(
    occa::dtype::double_.canBeCastedTo(occa::dtype::double3)
  );
  ASSERT_FALSE(
    occa::dtype::double2.canBeCastedTo(occa::dtype::double3)
  );
  ASSERT_FALSE(
    occa::dtype::double3.canBeCastedTo(occa::dtype::double2)
  );

  // double  <---> byte
  // double2 <---> byte
  ASSERT_TRUE(
    occa::dtype::double_.canBeCastedTo(occa::dtype::byte)
  );
  ASSERT_TRUE(
    occa::dtype::byte.canBeCastedTo(occa::dtype::double_)
  );
  ASSERT_TRUE(
    occa::dtype::double2.canBeCastedTo(occa::dtype::byte)
  );
  ASSERT_TRUE(
    occa::dtype::byte.canBeCastedTo(occa::dtype::double2)
  );
}

void testGet() {
  ASSERT_EQ(occa::dtype::float_,
            occa::dtype::get<float>());

  occa::dtypeVector types = occa::dtype::getMany<float, double, int>();
  ASSERT_EQ(3,
            (int) types.size());
  ASSERT_EQ(occa::dtype::float_,
            types[0]);
  ASSERT_EQ(occa::dtype::double_,
            types[1]);
  ASSERT_EQ(occa::dtype::int_,
            types[2]);
}

void testJsonMethods() {
  ASSERT_EQ(occa::dtype::toJson(occa::dtype::double_).toString(),
            occa::json::parse("{ type: 'builtin', name: 'double' }").toString());

  occa::dtype_t foo("foo");
  foo.addField("a", occa::dtype::double_)
    .addField("b", occa::dtype::double_);

  const std::string fooJsonStr = (
    "{"
    "  type: 'struct',"
    "  fields: ["
    "    { name: 'a', dtype: { type: 'builtin', name: 'double' } },"
    "    { name: 'b', dtype: { type: 'builtin', name: 'double' } },"
    "  ],"
    "}"
  );

  occa::json fooJson = occa::json::parse(fooJsonStr);
  ASSERT_EQ(fooJson.toString(),
            occa::dtype::toJson(foo).toString());

  occa::dtype_t foo2 = occa::dtype::fromJson(fooJsonStr);
  ASSERT_NEQ(foo, foo2);
  ASSERT_TRUE(foo.matches(foo2));
}
