#include <occa.hpp>
#include <occa/tools/testing.hpp>

using namespace occa;

template <class TM>
TM castNum(const int i) {
  return fromString<TM>(
    toString(i)
  );
}

template <class TM>
void testIndexOf() {
  std::vector<TM> vec;
  for (int i = 0; i < 10; ++i) {
    vec.push_back(castNum<TM>(i));
  }

  ASSERT_EQ(
    (dim_t) 5,
    indexOf(vec, castNum<TM>(5))
  );

  ASSERT_EQ(
    (dim_t) -1,
    indexOf(vec, castNum<TM>(-1))
  );

  ASSERT_EQ(
    (dim_t) -1,
    indexOf(vec, castNum<TM>(11))
  );
}

void testJoin();

int main(const int argc, const char **argv) {
  testIndexOf<int>();
  testIndexOf<float>();
  testIndexOf<std::string>();
  testJoin();

  return 0;
}

void testJoin() {
  strVector vec;
  vec.push_back("a");
  vec.push_back("b");
  vec.push_back("c");
  ASSERT_EQ(join(vec, ","),
            "a,b,c");
  ASSERT_EQ(join(vec, " , "),
            "a , b , c");
}
