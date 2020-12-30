#include <occa.hpp>

#include <occa/internal/utils/vector.hpp>
#include <occa/internal/utils/testing.hpp>

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

int main(const int argc, const char **argv) {
  testIndexOf<int>();
  testIndexOf<float>();
  testIndexOf<std::string>();

  return 0;
}
