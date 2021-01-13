#include <occa.hpp>
#include <occa/internal/utils/misc.hpp>
#include <occa/internal/utils/testing.hpp>

void testPtrMethods();

int main(const int argc, const char **argv) {
  testPtrMethods();

  return 0;
}

void testPtrMethods() {
  int a;
  int *b = new int[10];

  ASSERT_EQ(occa::ptrDiff(&a, &a), (occa::udim_t) 0);
  ASSERT_EQ(occa::ptrDiff(b, b)  , (occa::udim_t) 0);

  ASSERT_GT(occa::ptrDiff(&a, b), (occa::udim_t) 0);
  ASSERT_GT(occa::ptrDiff(b, &a), (occa::udim_t) 0);
}
