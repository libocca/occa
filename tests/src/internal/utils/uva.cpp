#include <occa/internal/utils/testing.hpp>

#include <occa.hpp>
#include <occa/internal/utils/uva.hpp>

void testPtrRange();

int main(const int argc, const char **argv) {
  testPtrRange();

  return 0;
}

void testPtrRange() {
  occa::ptrRange range = occa::ptrRange();
  ASSERT_EQ(range.start,
            (char*) NULL);
  ASSERT_EQ(range.end,
            (char*) NULL);

  range = occa::ptrRange((void*) 10,
                         10);
  // [10,20) = [10,20)
  ASSERT_EQ(range,
            occa::ptrRange((void*) 10,
                           10));
  // [10,20) !n [5, 10)
  ASSERT_NEQ(range,
             occa::ptrRange((void*) 5,
                            5));
  // [10,20) !n [20, 30)
  ASSERT_NEQ(range,
             occa::ptrRange((void*) 20,
                            10));
  // [10,20) n [15,25)
  ASSERT_EQ(range,
             occa::ptrRange((void*) 15,
                            10));
  // [10,20) !n [0,5)
  ASSERT_NEQ(range,
             occa::ptrRange((void*) 0,
                            5));
  // [10,20) !n [25,35)
  ASSERT_NEQ(range,
             occa::ptrRange((void*) 25,
                            10));

  // [10,20) == [11,16)
  ASSERT_FALSE(range < occa::ptrRange((void*) 11,
                                      5));
  // [10,20) == [11,21)
  ASSERT_FALSE(range < occa::ptrRange((void*) 11,
                                      10));
  // [10,20) > [0,5)
  ASSERT_FALSE(range < occa::ptrRange((void*) 0,
                                      5));
  // [10,20) < [21,31)
  ASSERT_TRUE(range < occa::ptrRange((void*) 21,
                                     10));

  std::cout << "Testing ptrRange output: " << range << '\n';
}
